#ifndef Application_hpp
#define Application_hpp

#include <iostream>
#include <string>
#include <deque>
#include "rock_util/LogReader.hpp"
#include "sonar_processing/TargetTrack.hpp"
#include "base/Plot.hpp"

namespace sonarlog_features {

class Application {
public:

    void init(const std::string& filename, const std::string& stream_name);

    void process_logfile();

    void process_next_sample();

    void plot(cv::Mat mat);

    base::Plot& plot() {
        return *(plot_.get());
    }

    static Application* instance();

private:

    Application() {}

    ~Application() {}

    std::auto_ptr<rock_util::LogReader> reader_;
    rock_util::LogStream stream_;


    static Application *instance_;
    std::auto_ptr<base::Plot> plot_;

    cv::Mat rls_w1, rls_w2, rls_p1, rls_p2;
    std::deque<cv::Mat> frames;
};

} /* namespace sonarlog_features */



#endif /* Application_hpp */
