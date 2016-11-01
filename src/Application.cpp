#include <opencv2/opencv.hpp>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "sonar_util/Converter.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonarlog_slam/Application.hpp"
#include "sonar_processing/ImageUtils.hpp"

using namespace sonar_processing;

namespace sonarlog_slam {

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
    uint32_t frame_height = 500;
    uint32_t frame_width = base::MathUtil::aspect_ratio_width(angle, frame_height);
    cv::Mat src = sonar_util::Converter::convert2polar(sample.bins, bearings, sample.bin_count, sample.beam_count, frame_width, frame_height);
    cv::imshow("src", src);
    cv::waitKey(100);
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
