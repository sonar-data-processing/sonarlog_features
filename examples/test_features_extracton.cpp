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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace sonar_processing;
using namespace sonar_processing::denoising;

inline void load_sonar_holder(const base::samples::Sonar& sample, sonar_processing::SonarHolder& sonar_holder) {
    sonar_holder.Reset(sample.bins,
        rock_util::Utilities::get_radians(sample.bearings),
        sample.beam_width.getRad(),
        sample.bin_count,
        sample.beam_count);
}

inline void peform_preprocessing(const cv::Mat& src, const cv::Mat& cart_roi_mask, cv::Mat& dst) {
    /* gradient */
    cv::Mat cart_image;
    src.convertTo(cart_image, CV_8U, 255);
    cv::Mat cart_aux, cart_grad;
    cv::boxFilter(cart_image, cart_aux, CV_8U, cv::Size(5, 5));
    preprocessing::gradient_filter(cart_aux, cart_grad);
    cv::normalize(cart_grad, cart_grad, 0, 255, cv::NORM_MINMAX);
    cv::boxFilter(cart_aux, cart_aux, CV_8U, cv::Size(30, 30));
    cart_grad -= cart_aux;
    cv::normalize(cart_grad, cart_grad, 0, 255, cv::NORM_MINMAX);
    cv::Mat cart_roi_mask2;
    cv::morphologyEx(cart_roi_mask, cart_roi_mask2, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
    cv::Mat cart_grad2;
    cart_grad.copyTo(cart_grad2, cart_roi_mask2);
    cv::medianBlur(cart_grad2, dst, 5);
}

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        // DATA_PATH_STRING + "/logs/gemini-harbor.3.log",
        "/home/gustavoneves/masters_degree/dataset/gemini/testday/gemini-harbor.3.log"
                // "/arquivos/Logs/gemini/dataset_gustavo/logs/20160316-1127-06925_07750-gemini.0.log",
    };

    uint num_logfiles = sizeof(logfiles) / sizeof(std::string);
    RLS rls(3);
    sonar_processing::SonarHolder sonar_holder;
    
    // int nfeatures = 100;
    // float scaleFactor = 1.4f;
    // int nlevels = 16;
    // int edgeThreshold = 31;
    // int firstLevel = 0;
    // int WTA_K=2;
    // int scoreType=cv::ORB::FAST_SCORE;
    // int patchSize=31;
    // 
    // cv::OrbFeatureDetector detector(
    //     nfeatures,
    //     scaleFactor,
    //     nlevels,
    //     edgeThreshold,
    //     firstLevel,
    //     WTA_K,
    //     scoreType,
    //     patchSize);
    // 
    // cv::OrbDescriptorExtractor descriptor;

    cv::Ptr<cv::FeatureDetector> detector =  cv::FeatureDetector::create("MSER");
    cv::Ptr<cv::DescriptorExtractor> descriptor =  cv::DescriptorExtractor::create("BRISK");

    for (size_t i = 0; i < num_logfiles; i++) {
        rock_util::LogReader reader(logfiles[i]);
        rock_util::LogStream stream = reader.stream("gemini.sonar_samples");

        base::samples::Sonar sample;
        cv::Mat last_cart_grad; 
        cv::Mat last_desc;
        std::vector<cv::KeyPoint> last_keypoints;

        while (stream.current_sample_index() < stream.total_samples()) {

            stream.next<base::samples::Sonar>(sample);
            load_sonar_holder(sample, sonar_holder);
            
            /* current frame */
            cv::Mat cart_raw = sonar_holder.cart_image();
            cv::resize(sonar_holder.cart_image(), cart_raw, cv::Size(), 0.5, 0.5);

            /* denoising */
            cv::Mat cart_denoised;
            cart_denoised = rls.sliding_window(cart_raw);
            
            /* roi masks */
            cv::Mat cart_roi_mask, polar_roi_mask, cart_image;
            preprocessing::extract_roi_masks(cart_denoised, sonar_holder.bearings(), sonar_holder.bin_count(), sonar_holder.beam_count(),
                                             cart_roi_mask, polar_roi_mask, 0.1);                                 
            cart_denoised.copyTo(cart_image, cart_roi_mask);

            cv::Mat cart_grad;
            peform_preprocessing(cart_image, cart_roi_mask, cart_grad);

            std::vector<cv::KeyPoint> keypoints;
            cv::Mat desc;            
            detector->detect(cart_grad, keypoints);
            descriptor->compute(cart_grad, keypoints, desc);
            
            for (int i = 0; i < keypoints.size(); i++) {
                std::cout << "keypoints.size()  " << keypoints[i].size << std::endl;
            }

            if (!last_cart_grad.empty()) {                
                cv::Mat canvas;
                std::vector<cv::DMatch> matches;
            
                cv::BFMatcher matcher(cv::NORM_L2, true);
                matcher.match(desc, last_desc, matches);
            
                std::cout << "keypoints: " << keypoints.size() << std::endl;
                std::cout << "last_keypoints: " << last_keypoints.size() << std::endl;
                cv::drawMatches(cart_grad, keypoints, last_cart_grad, last_keypoints, matches, canvas);                    
                cv::imshow("canvas", canvas);
                cv::waitKey(10);
            }
            
            last_keypoints.clear();
            last_keypoints = keypoints;

            last_desc = desc.clone();
            last_cart_grad = cart_grad.clone();

        }
    }
}
