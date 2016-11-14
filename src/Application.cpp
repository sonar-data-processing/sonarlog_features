#include <opencv2/opencv.hpp>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "sonar_util/Converter.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonarlog_features/Application.hpp"
#include "sonar_processing/ImageUtils.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_processing/Denoising.hpp"
#include "sonar_processing/QualityMetrics.hpp"

using namespace sonar_processing;

namespace sonarlog_features {

Application *Application::instance_ = NULL;

Application*  Application::instance() {
    if (!instance_){
        instance_ = new Application();
    }
    return instance_;
}

void Application::init(const std::string& filename, const std::string& stream_name) {
    reader_.reset(new rock_util::LogReader(filename));
    plot_.reset(new base::Plot());
    stream_ = reader_->stream(stream_name);
}

void Application::process_next_sample() {
    base::samples::Sonar sample;
    stream_.next<base::samples::Sonar>(sample);

    /* current frame */
    std::vector<float> bearings = rock_util::Utilities::get_radians(sample.bearings);
    float angle = bearings[bearings.size()-1];
    uint32_t frame_height = 400;
    uint32_t frame_width = base::MathUtil::aspect_ratio_width(angle, frame_height);
    cv::Mat input = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);

    /* roi frame */
    cv::Mat src(sample.beam_count, sample.bin_count, CV_32F, (void*) sample.bins.data());
    src.convertTo(src, CV_8U, 255);

    /* image enhancement */
    cv::Mat enhanced;
    preprocessing::adaptative_clahe(src, enhanced);

    /* denoising process */
    // denoising::homomorphic_filter(enhanced, denoised, 2);
    cv::Mat denoised1, denoised2;
    denoising::rls(rls_w1, rls_p1, enhanced, denoised1);
    denoising::rls_sliding_window(rls_w2, rls_p2, frames, 4, enhanced, denoised2);
    cv::Mat result;
    result.push_back(enhanced);
    result.push_back(denoised1);
    result.push_back(denoised2);
    cv::resize(result, result, cv::Size(result.cols * 0.8, result.rows * 0.8));
    cv::imshow("result", result);

    /* convert to cartesian plane */
    sample.bins.assign((float*) denoised2.datastart, (float*) denoised2.dataend);
    cv::Mat output = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);
    input.push_back(output);

    cv::imshow("output", input);
    cv::waitKey();
}

void Application::process_logfile() {
    stream_.reset();
    while (stream_.current_sample_index() < stream_.total_samples()) process_next_sample();
    cv::waitKey();
}

void Application::plot(cv::Mat mat) {
    (*plot_)(image_utils::mat2vector<float>(mat));
}

}
