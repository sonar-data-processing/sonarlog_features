#include <iostream>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_processing/Denoising.hpp"
#include "sonar_processing/ImageUtil.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_processing/SonarHolder.hpp"
#include "sonar_util/Converter.hpp"

#include <Eigen/Eigenvalues>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

using namespace sonar_processing;
using namespace sonar_processing::denoising;

inline void load_sonar_holder(const base::samples::Sonar& sample, sonar_processing::SonarHolder& sonar_holder) {
    sonar_holder.Reset(sample.bins,
        rock_util::Utilities::get_radians(sample.bearings),
        sample.beam_width.getRad(),
        sample.bin_count,
        sample.beam_count);
}

cv::Mat perform_preprocessing(const cv::Mat& src, cv::Mat cart_roi_mask) {
    /* gradient */
    cv::Mat cart_image, dst;
    src.convertTo(cart_image, CV_8U, 255);
    cv::Mat cart_aux, cart_grad;
    cv::boxFilter(cart_image, cart_aux, CV_8U, cv::Size(5, 5));
    preprocessing::gradient_filter(cart_aux, cart_grad);
    cv::normalize(cart_grad, cart_grad, 0, 255, cv::NORM_MINMAX);
    cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(30, 30));
    cart_grad -= cart_aux;
    cv::normalize(cart_grad, cart_grad, 0, 255, cv::NORM_MINMAX);

    /* mask */
    cv::morphologyEx(cart_roi_mask, cart_roi_mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
    cv::Mat cart_grad2;
    cart_grad.copyTo(cart_grad2, cart_roi_mask);
    cv::medianBlur(cart_grad2, cart_grad2, 5);

    /* threshold */
    cv::Mat cart_thresh;
    cv::threshold(cart_grad2, cart_thresh, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
    cv::morphologyEx(cart_thresh, cart_thresh, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)), cv::Point(-1, -1), 2);
    preprocessing::remove_blobs(cart_thresh, cart_thresh, cv::Size(8, 8));
    return cart_thresh;
}

void calculateCovarianceMatrix(std::vector<std::vector<cv::Point> > contours, cv::Mat reference_image, std::vector<cv::Mat>& covar_matrixes) {
    // first and second derivatives (horizontally and vertically)
    cv::Mat dx, dy, dxx, dyy;
    cv::Sobel(reference_image,  dx, CV_32F, 1, 0, 3);
    cv::Sobel(reference_image,  dy, CV_32F, 0, 1, 3);
    cv::Sobel(reference_image, dxx, CV_32F, 2, 0, 3);
    cv::Sobel(reference_image, dyy, CV_32F, 0, 2, 3);

    // magnitude and direction of gradients
    cv::Mat magnitude, direction;
    cv::magnitude(dx, dy, magnitude);
    cv::phase(dx, dy, direction, false);

    for (size_t i = 0; i < contours.size(); i++) {          // all blobs
        cv::Mat blob_description = cv::Mat::zeros(cv::Size(9, contours[i].size()), CV_32FC1);
        for (size_t j = 0; j < contours[i].size(); j++) {   // all pixels of same blob
            // positions
            blob_description.at<float>(j, 0) = contours[i][j].x;
            blob_description.at<float>(j, 1) = contours[i][j].y;
            // intensity
            blob_description.at<float>(j, 2) = reference_image.at<float>(contours[i][j].y, contours[i][j].x);
            // first and second derivatives
            blob_description.at<float>(j, 3) =  dx.at<float>(contours[i][j].y, contours[i][j].x);
            blob_description.at<float>(j, 4) =  dy.at<float>(contours[i][j].y, contours[i][j].x);
            blob_description.at<float>(j, 5) = dxx.at<float>(contours[i][j].y, contours[i][j].x);
            blob_description.at<float>(j, 6) = dyy.at<float>(contours[i][j].y, contours[i][j].x);
            // gradient magnitude and directions
            blob_description.at<float>(j, 7) = magnitude.at<float>(contours[i][j].y, contours[i][j].x);
            blob_description.at<float>(j, 8) = direction.at<float>(contours[i][j].y, contours[i][j].x);
        }
        // calculate covariance matrix
        cv::Mat covar, mean;
        cv::calcCovarMatrix(blob_description, covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
        covar = covar / (blob_description.rows - 1);
        covar_matrixes.push_back(covar);
    }
}

double calculateRiemannianDistance(cv::Mat A, cv::Mat B) {
    Eigen::MatrixXf e_A, e_B;
    cv::cv2eigen(A, e_A);
    cv::cv2eigen(B, e_B);

    /* compute generalized eigenvalues */
    Eigen::GeneralizedEigenSolver<Eigen::MatrixXf> ges;
    ges.compute(e_A, e_B);
    cv::Mat gev;
    cv::eigen2cv(Eigen::MatrixXf(ges.eigenvalues().real()), gev);

    /* riemannian distance */
    cv::Mat partial_res;
    cv::log(gev, partial_res);
    cv::pow(partial_res, 2, partial_res);
    double distance = sqrt(cv::sum(partial_res)[0]);
    return distance;
}

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/gemini-harbor.3.log",
        // "/home/gustavoneves/masters_degree/dataset/gemini/testday/gemini-harbor.3.log"
                // "/arquivos/Logs/gemini/dataset_gustavo/logs/20160316-1127-06925_07750-gemini.0.log",
    };

    uint num_logfiles = sizeof(logfiles) / sizeof(std::string);
    RLS rls(3);
    sonar_processing::SonarHolder sonar_holder;

    for (size_t i = 0; i < num_logfiles; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("gemini.sonar_samples");

        base::samples::Sonar sample;
        std::vector<cv::KeyPoint> last_keypoints;

        while (stream.current_sample_index() < stream.total_samples()) {

            stream.next<base::samples::Sonar>(sample);
            load_sonar_holder(sample, sonar_holder);

            /* current frame */
            cv::Mat cart_raw = sonar_holder.cart_image();
            cv::resize(cart_raw, cart_raw, cv::Size(), 0.5, 0.5);

            /* denoising */
            cv::Mat cart_denoised = rls.sliding_window(cart_raw);
            denoising::homomorphic_filter(cart_denoised, cart_denoised, 2);

            /* cartesian roi image */
            cv::Mat cart_drawable_area = sonar_holder.cart_image_mask();
            cv::resize(cart_drawable_area, cart_drawable_area, cart_denoised.size());
            cv::Mat cart_mask = preprocessing::extract_roi_mask(cart_denoised, cart_drawable_area, sonar_holder.bearings(), sonar_holder.bin_count(), sonar_holder.beam_count(), 0.1);
            cv::Mat cart_image;
            cart_denoised.copyTo(cart_image, cart_mask);

            /* preprocessing steps */
            cv::Mat cart_preprocessed = perform_preprocessing(cart_image, cart_mask);

            /* process all pixels inside each detected blob */
            std::vector<std::vector<cv::Point> > contours;
            cv::findContours(cart_preprocessed, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
            std::vector<cv::Mat> covar_matrixes;
            calculateCovarianceMatrix(contours, cart_denoised, covar_matrixes);

            /* comparison and matching */

            /* draw contours */
            cv::Mat dst = cv::Mat::zeros(cart_preprocessed.size(), CV_8UC3);
            cv::drawContours(dst, contours, -1, cv::Scalar(0,0,255), 2);
            cv::imshow("dst", dst);
            cv::waitKey();

        }
    }
}
