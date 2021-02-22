#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams();
    search_params = flannKsTreeSearchParams();
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat& query_desc,
                                 std::vector<std::vector<cv::DMatch>>& matches,
                                 int k) const
{
    cv::Mat indices(query_desc.rows, k, CV_32SC1);
    cv::Mat distances2(query_desc.rows, k, CV_32FC1);
    flann_index->knnSearch(query_desc, indices, distances2, k, *search_params);

    matches.resize(indices.rows);
    for (int i = 0; i < indices.rows; ++i) {
        matches[i] = std::vector<cv::DMatch>(k);
        for (int j = 0; j < k; ++j) {
            auto& match = matches[i][j];
            match.distance = std::sqrt(distances2.at<float>(i, j));
//            match.imgIdx =
            match.queryIdx = i;
            match.trainIdx = indices.at<int>(i, j);
        }
    }
}
