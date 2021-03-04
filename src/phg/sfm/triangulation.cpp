#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    int a_rows = 2 * count;
    int a_cols = 4;

    Eigen::MatrixXd A(a_rows, a_cols);

    for (int i = 0; i < count; ++i) {
        double x = ms[i][0];
        double y = ms[i][1];

        const auto& p0 = Ps[i].row(0);
        const auto& p1 = Ps[i].row(1);
        const auto& p2 = Ps[i].row(2);

        const auto r0 = x * p2 - p0;
        const auto r1 = y * p2 - p1;
        A.row(2 * i) << r0(0), r0(1), r0(2), r0(3);
        A.row(2 * i + 1) << r1(0), r1(1), r1(2), r1(3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd sol = svda.matrixV().transpose().row(a_cols - 1);

    return cv::Vec4d{sol[0], sol[1], sol[2], sol[3]};
}
