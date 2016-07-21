#include <opencv2/opencv.hpp>
#include <base/samples/Sonar.hpp>
#include "base/MathUtil.hpp"
#include "sonar_util/Converter.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "rock_util/Utilities.hpp"
#include "sonarlog_slam/Application.hpp"
#include "sonar_target_tracking/ImageUtils.hpp"

using namespace sonar_target_tracking;

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

    std::vector<float> bearings_radians = rock_util::Utilities::get_radians(sample.bearings);



    // TargetTrack target_track(sample.bins, bearings_radians, sample.beam_count, sample.bin_count);
    // target_track.apply();
    //
    //
    // cv::Rect roi = target_track.horiz_roi();
    //
    // cv::Mat mat = cv::Mat(sample.bins).reshape(1, sample.beam_count);
    //
    // cv::Mat dst = cv::Mat::zeros(mat.size(), mat.type());
    // mat(roi).copyTo(dst(roi));
    //
    // /* show result */
    // float angle = bearings_radians[bearings_radians.size()-1];
    // uint32_t frame_height = 600;
    // uint32_t frame_width = base::MathUtil::aspect_ratio_width(angle, frame_height);
    // std::vector<float> dst_bins = image_utils::mat2vector<float>(dst);
    // cv::Mat dst_polar = sonar_util::Converter::convert2polar(dst_bins, bearings_radians,
    //                                                          sample.bin_count, sample.beam_count,
    //                                                          frame_width, frame_height);

    // cv::imshow("dst_bins", dst_polar);
    // // cv::imshow("mat", dst);
    // cv::waitKey(100);

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
