#include <iostream>
#include <vector>
#include "sonarlog_features/Application.hpp"
#include "base/test_config.h"

using namespace sonarlog_features;

int main(int argc, char const *argv[]) {

    const std::string logfiles[] = {
        DATA_PATH_STRING + "/logs/gemini-ferry.0.log",
        DATA_PATH_STRING + "/logs/gemini-jequitaia.0.log",
    };

    for (int i = 0; i < 2; i++) {
        Application::instance()->init(logfiles[i], "gemini.sonar_samples");
        Application::instance()->process_next_sample();
    }

    return 0;
}
