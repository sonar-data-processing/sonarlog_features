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
    ORB_F,
    AKAZE_F
};

enum MatcherType {
    BRUTEFORCE_M,
    FLANN_M,
    RANSAC_M
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

    return cart_thresh.clone();
}

void extractAndDecribeFeatures(cv::Mat input, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, FeatureType type) {
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    switch(type) {
        case SIFT_F: {
                detector = cv::FeatureDetector::create("SIFT");
                detector->detect(input, keypoints);

                descriptor_extractor = cv::DescriptorExtractor::create("SIFT");
                descriptor_extractor->compute(input, keypoints, descriptors);
            }
            break;

        case SURF_F: {
                detector = new cv::SurfFeatureDetector();
                detector->detect(input, keypoints);

                descriptor_extractor = cv::DescriptorExtractor::create("SURF");
                descriptor_extractor->compute(input, keypoints, descriptors);
            }
            break;

        case ORB_F: {
                detector = cv::FeatureDetector::create("ORB");
                detector->detect(input, keypoints);

                descriptor_extractor = cv::DescriptorExtractor::create("ORB");
                descriptor_extractor->compute(input, keypoints, descriptors);
            }
            break;

        case AKAZE_F: {
                detector = cv::FeatureDetector::create("AKAZE");
                detector->detect(input, keypoints);

                descriptor_extractor = cv::DescriptorExtractor::create("AKAZE");
                descriptor_extractor->compute(input, keypoints, descriptors);
            }
        default:
            throw std::invalid_argument("Feature type parameter does not match a known enum value");
            break;
    }
}

std::vector<cv::DMatch> crossCheck(std::vector<cv::DMatch> matches12, std::vector<cv::DMatch> matches21) {
    std::vector<cv::DMatch> filteredMatches;
    for (size_t i = 0; i < matches12.size(); i++) {
        cv::DMatch forward = matches12[i];
        cv::DMatch backward = matches21[forward.trainIdx];
        if (backward.trainIdx == forward.queryIdx) {
            filteredMatches.push_back(forward);
        }
    }
    return filteredMatches;
}

cv::Mat matchFeatures(cv::Mat current_frame, std::vector<cv::KeyPoint> current_keypoints, cv::Mat current_descriptors,
                      cv::Mat last_frame, std::vector<cv::KeyPoint> last_keypoints, cv::Mat last_descriptors,
                      MatcherType type) {

    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::vector<cv::DMatch> matches12, matches21, filteredMatches;
    cv::Mat output;

    switch(type) {
        case BRUTEFORCE_M: {
            matcher = new cv::BFMatcher(cv::NORM_L1, true);
            matcher->match(current_descriptors, last_descriptors, matches12);

            if (matches12.size() < 4) {
                cv::drawMatches(current_frame, current_keypoints, last_frame, last_keypoints, matches12, output);
                return output;
            }

            std::cout << matches12.size() << std::endl;

            std::vector<cv::Point2f> obj, scene;
            for (size_t i = 0; i < matches12.size(); i++) {
                obj.push_back(current_keypoints[matches12[i].queryIdx].pt);
                scene.push_back(last_keypoints[matches12[i].trainIdx].pt);
            }

            std::cout << obj.size() << "," << scene.size() << std::endl;

            cv::Mat selectedSet;
            cv::Mat homography = cv::findHomography(obj, scene, CV_RANSAC, 0, selectedSet);

            std::vector<cv::DMatch> final_matchers;
            for (uint i = 0; i < selectedSet.rows; ++i) {
                if (selectedSet.at<int>(i))
                    final_matchers.push_back(matches12[i]);
            }

            cv::drawMatches(current_frame, current_keypoints, last_frame, last_keypoints, final_matchers, output);
            break;
        }

        case FLANN_M:
            // matcher = cv::DescriptorMatcher::create("Flann");
            matcher = new cv::FlannBasedMatcher();
            matcher->match(current_descriptors, last_descriptors, matches12);
            cv::drawMatches(current_frame, current_keypoints, last_frame, last_keypoints, matches12, output);
            break;

        default:
            throw std::invalid_argument("Matcher type parameter does not match a known enum value");
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

    cv::Mat last_frame;
    std::vector<cv::KeyPoint> last_keypoints;
    cv::Mat last_descriptors;

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
            cv::imshow("cart_grad2", cart_grad2);

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

            /* feature extraction amd description*/
            cv::initModule_nonfree();
            cv::Mat frame;
            cart_grad2.copyTo(frame);
            cv::imshow("frame", frame);

            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            extractAndDecribeFeatures(frame.clone(), keypoints, descriptors, SIFT_F);

            cv::Mat output;
            cv::cvtColor(frame, output, CV_GRAY2RGB);
            cv::drawKeypoints(frame, keypoints, output, cv::Scalar::all(-1), 4);
            cv:imshow("output", output);

            /* if it is not the first frame, match with previous one */
            if (!last_frame.empty()) {
                cv::Mat output = matchFeatures(frame.clone(), keypoints, descriptors.clone(),
                                               last_frame.clone(), last_keypoints, last_descriptors.clone(),
                                               BRUTEFORCE_M);
                cv::imshow("output2", output);
            //
            //     //
            //     // cv::Mat output2 = matchFeatures2(frame.clone(), keypoints, descriptors.clone(),
            //     //                                last_frame.clone(), last_keypoints, last_descriptors.clone(),
            //     //                                FLANN_M);
            //     // cv::imshow("output2", output2);
                cv::waitKey(0);
            }

            // store previous frame data
            frame.copyTo(last_frame);
            descriptors.copyTo(last_descriptors);
            last_keypoints = keypoints;
        }
    }
}
