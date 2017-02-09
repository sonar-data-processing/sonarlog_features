#include <iostream>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "base/test_config.h"
#include "rock_util/LogReader.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonar_processing/Clustering.hpp"
#include "sonar_processing/Denoising.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_util/Converter.hpp"
#include "sonarlog_features/Application.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "sonarlog_features/third_party/akaze/akaze_features.h"

using namespace sonarlog_features;
using namespace sonar_processing;
using namespace sonar_processing::denoising;

enum FeatureType {
    SIFT_F,
    SURF_F,
    FAST_F,
    ORB_F,
    BRISK_F,
    MSER_F,
    AKAZE_F,
    LBP_F
};

cv::Mat getMask(cv::Mat cart_image, cv::Mat cart_roi_mask) {
    /* gradient */
    cart_image.convertTo(cart_image, CV_8U, 255);
    cv::Mat cart_aux, cart_grad;
    cv::boxFilter(cart_image, cart_aux, CV_8U, cv::Size(5, 5));
    preprocessing::gradient_filter(cart_aux, cart_grad);
    cv::normalize(cart_grad, cart_grad, 0, 255, cv::NORM_MINMAX);
    cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(30, 30));
    cart_grad -= cart_aux;
    cv::normalize(cart_grad, cart_grad, 0, 255, cv::NORM_MINMAX);
    cv::morphologyEx(cart_roi_mask, cart_roi_mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
    cv::Mat cart_grad2;
    cart_grad.copyTo(cart_grad2, cart_roi_mask);
    cv::medianBlur(cart_grad2, cart_grad2, 5);

    /* filtering */
    cv::Mat cart_filtered;
    cv::blur(cart_grad2, cart_filtered, cv::Size(5, 5));
    for (size_t i = 0; i < cart_filtered.rows; i++) {
        for (size_t j = 0; j < cart_filtered.cols; j++) {
            if (cart_filtered.at<uchar>(i,j) < 255 * 0.2)
            cart_filtered.at<uchar>(i,j) = 0;
        }
    }

    /* threshold */
    cv::Mat cart_thresh;
    cv::threshold(cart_filtered, cart_thresh, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
    cv::morphologyEx(cart_roi_mask, cart_roi_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
    preprocessing::remove_blobs(cart_thresh, cart_thresh, cv::Size(5, 5));

    // cart_thresh.convertTo(cart_thresh, CV_32F, 1.0 / 255);
    // cart_grad2.convertTo(cart_grad2, CV_32F, 1.0 / 255.0);
    return cart_thresh.clone();
}

cv::Mat extractFeatures(cv::Mat input, FeatureType type) {

    cv::Ptr<cv::FeatureDetector> detector;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat output;
    switch(type) {
        case SIFT_F: {
                detector = cv::FeatureDetector::create("SIFT");
                detector->detect(input, keypoints);
            }
            cv::drawKeypoints(input, keypoints, output);
            break;
        case SURF_F: {
                detector = cv::FeatureDetector::create("SURF");
                detector->detect(input, keypoints);
                cv::drawKeypoints(input, keypoints, output);
            }
            break;
        case FAST_F: {
                detector = cv::FeatureDetector::create("FAST");
                detector->detect(input, keypoints);
                cv::drawKeypoints(input, keypoints, output);
            }
        break;
        case ORB_F: {
                detector = cv::FeatureDetector::create("FAST");
                detector->detect(input, keypoints);
                cv::drawKeypoints(input, keypoints, output);
            }
            break;
        case BRISK_F: {
                detector = cv::FeatureDetector::create("BRISK");
                detector->detect(input, keypoints);
                cv::drawKeypoints(input, keypoints, output);
            }
            break;
        case MSER_F: {
                detector = cv::FeatureDetector::create("MSER");
                detector->detect(input, keypoints);
                cv::drawKeypoints(input, keypoints, output);
            }
            break;
        case AKAZE_F: {
                detector = cv::FeatureDetector::create("AKAZE");
                detector->detect(input, keypoints);
                cv::drawKeypoints(input, keypoints, output);
            }
            break;
        // case LBP_F: {
        //         detector = cv::FeatureDetector::create("FREAK");
        //         detector->detect(input, keypoints);
        //         cv::drawKeypoints(input, keypoints, output);
        //     }
        //     break;
        default:
            break;
    }

    return output.clone();
}

int main(int argc, char const *argv[]) {

const std::string logfiles[] = {
    DATA_PATH_STRING + "/logs/gemini-harbor.3.log",
            // "/arquivos/Logs/gemini/dataset_gustavo/logs/20160316-1127-06925_07750-gemini.0.log",
};

uint num_logfiles = sizeof(logfiles) / sizeof(std::string);
RLS rls(3);


for (size_t i = 0; i < num_logfiles; i++) {
    rock_util::LogReader reader(logfiles[i]);
    rock_util::LogStream stream = reader.stream("gemini.sonar_samples");

    base::samples::Sonar sample;
    cv::Mat last_frame;
    while (stream.current_sample_index() < stream.total_samples()) {

        stream.next<base::samples::Sonar>(sample);

        /* cartesian properties */
        std::vector<float> bearings = rock_util::Utilities::get_radians(sample.bearings);
        float angle = bearings[bearings.size() - 1];
        uint32_t frame_height = 400;
        uint32_t frame_width = base::MathUtil::aspect_ratio_width(angle, frame_height);

        /* current frame */
        cv::Mat cart_raw = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);
        cart_raw.convertTo(cart_raw, CV_32F, 1.0 / 255);

        /* denoising */
        cv::Mat cart_denoised;
        cart_denoised = rls.sliding_window(cart_raw);

        /* roi masks */
        cv::Mat cart_roi_mask, polar_roi_mask, cart_image;
        preprocessing::extract_roi_masks(cart_denoised, bearings, sample.bin_count, sample.beam_count, cart_roi_mask, polar_roi_mask, 0.1);
        cart_denoised.copyTo(cart_image, cart_roi_mask);

        /* gradient */
        cart_image.convertTo(cart_image, CV_8U, 255);
        cv::Mat cart_aux, cart_grad;
        cv::boxFilter(cart_image, cart_aux, CV_8U, cv::Size(5, 5));
        preprocessing::gradient_filter(cart_aux, cart_grad);
        cv::normalize(cart_grad, cart_grad, 0, 255, cv::NORM_MINMAX);
        cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(30, 30));
        cart_grad -= cart_aux;
        cv::normalize(cart_grad, cart_grad, 0, 255, cv::NORM_MINMAX);
        cv::morphologyEx(cart_roi_mask, cart_roi_mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
        cv::Mat cart_grad2;
        cart_grad.copyTo(cart_grad2, cart_roi_mask);
        cv::medianBlur(cart_grad2, cart_grad2, 5);

        /* filtering */
        cv::Mat cart_filtered;
        cv::blur(cart_grad2, cart_filtered, cv::Size(5, 5));
        for (size_t i = 0; i < cart_filtered.rows; i++) {
            for (size_t j = 0; j < cart_filtered.cols; j++) {
                if (cart_filtered.at<uchar>(i,j) < 255 * 0.2)
                cart_filtered.at<uchar>(i,j) = 0;
            }
        }

        /* threshold */
        cv::Mat cart_thresh;
        cv::threshold(cart_filtered, cart_thresh, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
        cv::morphologyEx(cart_roi_mask, cart_roi_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
        preprocessing::remove_blobs(cart_thresh, cart_thresh, cv::Size(5, 5));

        /* feature extraction */
        cv::initModule_nonfree();

        cv::Mat sift_output  = extractFeatures(cart_grad2.clone(), SIFT_F);
        cv::Mat surf_output  = extractFeatures(cart_grad2.clone(), SURF_F);
        cv::Mat fast_output  = extractFeatures(cart_grad2.clone(), FAST_F);
        cv::Mat orb_output   = extractFeatures(cart_grad2.clone(), ORB_F);
        cv::Mat brisk_output = extractFeatures(cart_grad2.clone(), BRISK_F);
        cv::Mat mser_output  = extractFeatures(cart_grad2.clone(), MSER_F);
        cv::Mat akaze_output = extractFeatures(cart_grad2.clone(), AKAZE_F);
        cv::Mat lbp_output   = extractFeatures(cart_grad2.clone(), LBP_F);

        cv::Mat output1, output2, output;
        cv::hconcat(sift_output, surf_output, output1);
        cv::hconcat(output1, fast_output, output1);
        cv::hconcat(output1, orb_output, output1);

        cv::hconcat(brisk_output, mser_output, output2);
        cv::hconcat(output2, akaze_output, output2);
        cv::hconcat(output2, akaze_output, output2);

        cv::vconcat(output1, output2, output);
        cv::resize(output, output, cv::Size(output.cols * 0.5, output.rows * 0.5));
        cv::imshow("output", output);
        cv::waitKey(30);
    }
    cv::waitKey(0);
}

return 0;
}
