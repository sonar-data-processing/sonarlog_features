#include <iostream>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char const *argv[]) {

    cv::Mat_<float> samples1 = (cv::Mat_<float>(3, 2) << 500.0, 350.2,
                                                        500.5, 355.8,
                                                        498.7, 352.0);
    cv::Mat_<float> samples2 = (cv::Mat_<float>(3, 2) << 500.0, 350.2,
                                                        498.7, 352.0,
                                                        550.0, 348.0);

    cv::Mat cov1, mu1;
    cv::calcCovarMatrix(samples1, cov1, mu1, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    cov1 = cov1 / (samples1.rows - 1);
    std::cout << "Covariance Matrix A: " << cov1 << std::endl;
    //      MATLAB OUTPUT
    //      ----------------
    //      0.8633    1.2167
    //      1.2167    8.1733

    cv::Mat cov2, mu2;
    cv::calcCovarMatrix(samples2, cov2, mu2, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    cov2 = cov2 / (samples2.rows - 1);
    std::cout << "Covariance Matrix B: " << cov2 << std::endl;
    //      MATLAB OUTPUT:
    //      ----------------
    //      855.5633  -52.9233
    //      -52.9233    4.0133

    /* compute generalized eigenvalues */
    Eigen::MatrixXf e_A, e_B;
    cv::cv2eigen(cov1, e_A);
    cv::cv2eigen(cov2, e_B);

    Eigen::GeneralizedEigenSolver<Eigen::MatrixXf> ges;
    ges.compute(e_A, e_B);
    cv::Mat gev;
    cv::eigen2cv(Eigen::MatrixXf(ges.eigenvalues().real()), gev);
    std::cout << "Generalized eigenvalues: " << ges.eigenvalues().real() << std::endl;
    //      MATLAB OUTPUT:
    //      ----------------
    //      0.0008
    //      11.2591

    /* calculate riemannian distance */
    cv::Mat partial_res;
    cv::log(gev, partial_res);
    cv::pow(partial_res, 2, partial_res);
    double distance = sqrt(cv::sum(partial_res)[0]);
    std::cout << "Riemannian distance: " << distance << std::endl;
    //      MATLAB OUTPUT:
    //      ----------------
    //      7.5515
    return 0;
}
