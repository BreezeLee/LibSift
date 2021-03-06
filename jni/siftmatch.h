#ifndef SIFTMATCH_H
#define SIFTMATCH_H

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

namespace Ui {
class SiftMatch;
}

class SiftMatch
{
    
public:
    ~SiftMatch();

    void CalcFourCorner();//计算图2的四个角经矩阵H变换后的坐标
    

private:
	void initial();

    int open_image_number;//打开图片个数
    string name1,name2;//两张图片的文件名
    IplImage *img1, *img2;//IplImage格式的原图
    IplImage *img1_Feat, *img2_Feat;//画上特征点之后的图

    bool verticalStackFlag;//显示匹配结果的合成图像中，两张图是纵向排列的标志
    IplImage *stacked;//显示匹配结果的合成图像，显示经距离比值法筛选后的匹配结果
    IplImage *stacked_ransac;//显示匹配结果的合成图像，显示经RANSAC算法筛选后的匹配结果

    struct feature *feat1, *feat2;//feat1：图1的特征点数组，feat2：图2的特征点数组
    int n1, n2;//n1:图1中的特征点个数，n2：图2中的特征点个数
    struct feature *feat;//每个特征点
    struct kd_node *kd_root;//k-d树的树根
    struct feature **nbrs;//当前特征点的最近邻点数组

    CvMat * H;//RANSAC算法求出的变换矩阵
    struct feature **inliers;//精RANSAC筛选后的内点数组
    int n_inliers;//经RANSAC算法筛选后的内点个数,即feat2中具有符合要求的特征点的个数

    IplImage *xformed;//临时拼接图，即只将图2变换后的图
    IplImage *xformed_simple;//简易拼接图
    IplImage *xformed_proc;//处理后的拼接图

//    int img1LeftBound;//图1中匹配点外接矩形的左边界
//    int img1RightBound;//图1中匹配点外接矩形的右边界
//    int img2LeftBound;//图2中匹配点外接矩形的左边界
//    int img2RightBound;//图2中匹配点外接矩形的右边界

    //图2的四个角经矩阵H变换后的坐标
    CvPoint leftTop,leftBottom,rightTop,rightBottom;


};

#endif // SIFTMATCH_H
