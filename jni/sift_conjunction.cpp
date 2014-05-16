#include "siftmatch.h"
#include "com_LibSift_namespace_LibSiftActivity.h"
#include <iostream>
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
using namespace cv;

extern "C"
{
#include "imgfeatures.h"
#include "kdtree.h"
#include "minpq.h"
#include "sift.h"
#include "utils.h"
#include "xform.h"
}

//在k-d树上进行BBF搜索的最大次数
/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

//目标点与最近邻和次近邻的距离的比值的阈值，若大于此阈值，则剔除此匹配点对
//通常此值取0.6，值越小找到的匹配点对越精确，但匹配数目越少
/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

//窗口名字符串
#define IMG1 "图1"
#define IMG2 "图2"
#define IMG1_FEAT "图1特征点"
#define IMG2_FEAT "图2特征点"
#define IMG_MATCH1 "距离比值筛选后的匹配结果"
#define IMG_MATCH2 "RANSAC筛选后的匹配结果"
#define IMG_MOSAIC_TEMP "临时拼接图像"
#define IMG_MOSAIC_SIMPLE "简易拼接图"
#define IMG_MOSAIC_BEFORE_FUSION "重叠区域融合前"
#define IMG_MOSAIC_PROC "处理后的拼接图"

JNIEXPORT jint JNICALL Java_com_LibSift_namespace_SiftFun_siftConjunction
(JNIEnv * env, jclass cls, jstring jfilename1, jstring jfilename2)
{
	const char* filename1 = env->GetStringUTFChars(jfilename1, NULL);
    const char* filename2 = env->GetStringUTFChars(jfilename2, NULL);
	
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

    //图2的四个角经矩阵H变换后的坐标
    CvPoint leftTop,leftBottom,rightTop,rightBottom;
	string img_name1 = filename1;
	string img_name2 = filename2;
	img1 = cvLoadImage(img_name1.c_str());
	img2 = cvLoadImage(img_name2.c_str());
	img1_Feat = cvCloneImage(img1);//复制图1，深拷贝，用来画特征点
    img2_Feat = cvCloneImage(img2);//复制图2，深拷贝，用来画特征点
	n1 = sift_features( img1, &feat1 );//检测图1中的SIFT特征点,n1是图1的特征点个数
    export_features("feature1.txt",feat1,n1);//将特征向量数据写入到文件
    draw_features( img1_Feat, feat1, n1 );//画出特征点
	string name1_Feat = img_name1;//文件名，原文件名加"_Feat"
	//cvSaveImage(name1_Feat.insert( name1_Feat.find_last_of(".",-1) , "_Feat").c_str(),img1_Feat);//保存图片
	n2 = sift_features( img2, &feat2 );//检测图2中的SIFT特征点，n2是图2的特征点个数
    export_features("feature2.txt",feat2,n2);//将特征向量数据写入到文件
    draw_features( img2_Feat, feat2, n2 );//画出特征点
	string name2_Feat = img_name2;
    //cvSaveImage(name2_Feat.insert( name2_Feat.find_last_of(".",-1) , "_Feat").c_str(),img2_Feat);//保存图片
	

	stacked = stack_imgs_horizontal(img1, img2);
	kd_root = kdtree_build( feat1, n1 );
	Point pt1,pt2;//连线的两个端点
    double d0,d1;//feat2中每个特征点到最近邻和次近邻的距离
    int matchNum = 0;//经距离比值法筛选后的匹配点对的个数

    //遍历特征点集feat2，针对feat2中每个特征点feat，选取符合距离比值条件的匹配点，放到feat的fwd_match域中
    for(int i = 0; i < n2; i++ )
    {
        feat = feat2+i;//第i个特征点的指针
        //在kd_root中搜索目标点feat的2个最近邻点，存放在nbrs中，返回实际找到的近邻点个数
        int k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
        if( k == 2 )
        {
            d0 = descr_dist_sq( feat, nbrs[0] );//feat与最近邻点的距离的平方
            d1 = descr_dist_sq( feat, nbrs[1] );//feat与次近邻点的距离的平方
            //若d0和d1的比值小于阈值NN_SQ_DIST_RATIO_THR，则接受此匹配，否则剔除
            if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
            {   //将目标点feat和最近邻点作为匹配点对
                pt2 = Point( cvRound( feat->x ), cvRound( feat->y ) );//图2中点的坐标
                pt1 = Point( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );//图1中点的坐标(feat的最近邻点)
                pt2.x += img1->width;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点
                cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );//画出连线
                matchNum++;//统计匹配点对的个数
                feat2[i].fwd_match = nbrs[0];//使点feat的fwd_match域指向其对应的匹配点
            }
        }
        free( nbrs );//释放近邻数组
    }
	String name_match_DistRatio = img_name1;//文件名，原文件名去掉序号后加"_match_DistRatio"
	//cvSaveImage(name_match_DistRatio.replace( name_match_DistRatio.find_last_of(".",-1)-1 , 1 , "_match_DistRatio").c_str(),stacked);


	 //利用RANSAC算法筛选匹配点,计算变换矩阵H，
    //无论img1和img2的左右顺序，H永远是将feat2中的特征点变换为其匹配点，即将img2中的点变换为img1中的对应点
    H = ransac_xform(feat2,n2,FEATURE_FWD_MATCH,lsq_homog,4,0.01,homog_xfer_err,3.0,&inliers,&n_inliers);

    //若能成功计算出变换矩阵，即两幅图中有共同区域
    if( H )
    {
		std::cout<<"经RANSAC算法筛选后的匹配点对个数："<<n_inliers<<std::endl;
        int invertNum = 0;//统计pt2.x > pt1.x的匹配点对的个数，来判断img1中是否右图

        //遍历经RANSAC算法筛选后的特征点集合inliers，找到每个特征点的匹配点，画出连线
        for(int i=0; i<n_inliers; i++)
        {
            feat = inliers[i];//第i个特征点
            pt2 = Point(cvRound(feat->x), cvRound(feat->y));//图2中点的坐标
            pt1 = Point(cvRound(feat->fwd_match->x), cvRound(feat->fwd_match->y));

            //统计匹配点的左右位置关系，来判断图1和图2的左右位置关系
            if(pt2.x > pt1.x)
                invertNum++;
            pt2.x += img1->width;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点
            cvLine(stacked_ransac,pt1,pt2,CV_RGB(255,0,255),1,8,0);//在匹配图上画出连线
        }

		String name_match_RANSAC = img_name1;//文件名，原文件名去掉序号后加"_match_RANSAC"

		/*程序中计算出的变换矩阵H用来将img2中的点变换为img1中的点，正常情况下img1应该是左图，img2应该是右图。
          此时img2中的点pt2和img1中的对应点pt1的x坐标的关系基本都是：pt2.x < pt1.x
          若用户打开的img1是右图，img2是左图，则img2中的点pt2和img1中的对应点pt1的x坐标的关系基本都是：pt2.x > pt1.x
          所以通过统计对应点变换前后x坐标大小关系，可以知道img1是不是右图。
          如果img1是右图，将img1中的匹配点经H的逆阵H_IVT变换后可得到img2中的匹配点*/

        //若pt2.x > pt1.x的点的个数大于内点个数的80%，则认定img1中是右图
        if(invertNum > n_inliers * 0.8)
        {
           std::cout<<"img1中是右图"<<std::endl;
            CvMat * H_IVT = cvCreateMat(3, 3, CV_64FC1);//变换矩阵的逆矩阵
            //求H的逆阵H_IVT时，若成功求出，返回非零值
            if( cvInvert(H,H_IVT) )
            {
                cvReleaseMat(&H);//释放变换矩阵H，因为用不到了
                H = cvCloneMat(H_IVT);//将H的逆阵H_IVT中的数据拷贝到H中
                cvReleaseMat(&H_IVT);//释放逆阵H_IVT
                //将img1和img2对调
                IplImage * temp = img2;
                img2 = img1;
                img1 = temp;
       
            }
            else//H不可逆时，返回0
            {
                cvReleaseMat(&H_IVT);//释放逆阵H_IVT
                std::cout<<"警告"<<"变换矩阵H不可逆"<<std::endl;
            }
        }
        else
		{
			;
		}
    }
    else //无法计算出变换矩阵，即两幅图中没有重合区域
    {
       std::cout<<"警告"<<"两图中无公共区域"<<std::endl;
    }



	 //若能成功计算出变换矩阵，即两幅图中有共同区域，才可以进行全景拼接
    if(H)
    {
        //拼接图像，img1是左图，img2是右图
        //计算图2的四个角经矩阵H变换后的坐标
		double v2[]={0,0,1};//左上角
		double v1[3];//变换后的坐标值
		CvMat V2 = cvMat(3,1,CV_64FC1,v2);
		CvMat V1 = cvMat(3,1,CV_64FC1,v1);
		cvGEMM(H,&V2,1,0,1,&V1);//矩阵乘法
		leftTop.x = cvRound(v1[0]/v1[2]);
		leftTop.y = cvRound(v1[1]/v1[2]);

		//将v2中数据设为左下角坐标
		v2[0] = 0;
		v2[1] = img2->height;
		V2 = cvMat(3,1,CV_64FC1,v2);
		V1 = cvMat(3,1,CV_64FC1,v1);
		cvGEMM(H,&V2,1,0,1,&V1);
		leftBottom.x = cvRound(v1[0]/v1[2]);
		leftBottom.y = cvRound(v1[1]/v1[2]);

		//将v2中数据设为右上角坐标
		v2[0] = img2->width;
		v2[1] = 0;
		V2 = cvMat(3,1,CV_64FC1,v2);
		V1 = cvMat(3,1,CV_64FC1,v1);
		cvGEMM(H,&V2,1,0,1,&V1);
		rightTop.x = cvRound(v1[0]/v1[2]);
		rightTop.y = cvRound(v1[1]/v1[2]);

		//将v2中数据设为右下角坐标
		v2[0] = img2->width;
		v2[1] = img2->height;
		V2 = cvMat(3,1,CV_64FC1,v2);
		V1 = cvMat(3,1,CV_64FC1,v1);
		cvGEMM(H,&V2,1,0,1,&V1);
		rightBottom.x = cvRound(v1[0]/v1[2]);
		rightBottom.y = cvRound(v1[1]/v1[2]);
        //为拼接结果图xformed分配空间,高度为图1图2高度的较小者，根据图2右上角和右下角变换后的点的位置决定拼接图的宽度
        xformed = cvCreateImage(cvSize(MIN(rightTop.x,rightBottom.x),MIN(img1->height,img2->height)),IPL_DEPTH_8U,3);
        //用变换矩阵H对右图img2做投影变换(变换后会有坐标右移)，结果放到xformed中
        cvWarpPerspective(img2,xformed,H,CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,cvScalarAll(0));

        //简易拼接法：直接将将左图img1叠加到xformed的左边
        xformed_simple = cvCloneImage(xformed);//简易拼接图，可笼子xformed
        cvSetImageROI(xformed_simple,cvRect(0,0,img1->width,img1->height));
        cvAddWeighted(img1,1,xformed_simple,0,0,xformed_simple);
        cvResetImageROI(xformed_simple);

        //处理后的拼接图，克隆自xformed
        xformed_proc = cvCloneImage(xformed);

        //重叠区域左边的部分完全取自图1
        cvSetImageROI(img1,cvRect(0,0,MIN(leftTop.x,leftBottom.x),xformed_proc->height));
        cvSetImageROI(xformed,cvRect(0,0,MIN(leftTop.x,leftBottom.x),xformed_proc->height));
        cvSetImageROI(xformed_proc,cvRect(0,0,MIN(leftTop.x,leftBottom.x),xformed_proc->height));
        cvAddWeighted(img1,1,xformed,0,0,xformed_proc);
        cvResetImageROI(img1);
        cvResetImageROI(xformed);
        cvResetImageROI(xformed_proc);
        //cvNamedWindow(IMG_MOSAIC_BEFORE_FUSION);
        //cvShowImage(IMG_MOSAIC_BEFORE_FUSION,xformed_proc);//显示融合之前的拼接图

        //采用加权平均的方法融合重叠区域
        int start = MIN(leftTop.x,leftBottom.x) ;//开始位置，即重叠区域的左边界
        double processWidth = img1->width - start;//重叠区域的宽度
        double alpha = 1;//img1中像素的权重
        for(int i=0; i<xformed_proc->height; i++)//遍历行
        {
            const uchar * pixel_img1 = ((uchar *)(img1->imageData + img1->widthStep * i));//img1中第i行数据的指针
            const uchar * pixel_xformed = ((uchar *)(xformed->imageData + xformed->widthStep * i));//xformed中第i行数据的指针
            uchar * pixel_xformed_proc = ((uchar *)(xformed_proc->imageData + xformed_proc->widthStep * i));//xformed_proc中第i行数据的指针
            for(int j=start; j<img1->width; j++)//遍历重叠区域的列
            {
                //如果遇到图像xformed中无像素的黑点，则完全拷贝图1中的数据
                if(pixel_xformed[j*3] < 50 && pixel_xformed[j*3+1] < 50 && pixel_xformed[j*3+2] < 50 )
                {
                    alpha = 1;
                }
                else
                {   //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比
                    alpha = (processWidth-(j-start)) / processWidth ;
                }
                pixel_xformed_proc[j*3] = pixel_img1[j*3] * alpha + pixel_xformed[j*3] * (1-alpha);//B通道
                pixel_xformed_proc[j*3+1] = pixel_img1[j*3+1] * alpha + pixel_xformed[j*3+1] * (1-alpha);//G通道
                pixel_xformed_proc[j*3+2] = pixel_img1[j*3+2] * alpha + pixel_xformed[j*3+2] * (1-alpha);//R通道
            }
        }

        //想通过锐化解决变换后的图像失真的问题，对于扭曲过大的图像，效果不好
        //double a[]={  0, -1,  0, -1,  5, -1, 0, -1,  0  };//拉普拉斯滤波核的数据
        //CvMat kernel = cvMat(3,3,CV_64FC1,a);//拉普拉斯滤波核
        //cvFilter2D(xformed_proc,xformed_proc,&kernel);//滤波
        //cvShowImage(IMG_MOSAIC_PROC,xformed_proc);//显示处理后的拼接图

        //保存拼接图
        string name_xformed = img_name1;//文件名，原文件名去掉序号后加"_Mosaic"
		//cvSaveImage(name_xformed.replace( name_xformed.find_last_of(".",-1)-1 , 1 , "_Mosaic").c_str(), xformed_simple);//保存简易拼接图
        cvSaveImage(name_xformed.insert( name_xformed.find_last_of(".",-1) , "_Proc").c_str(), xformed_proc);//保存处理后的拼接图
    }
	if(stacked)
    {
        kdtree_release(kd_root);//释放kd树
    }
	if( H )
    {
        cvReleaseMat(&H);//释放变换矩阵H
        free(inliers);//释放内点数组

        //释放RANSAC算法筛选后的匹配图
        cvReleaseImage(&stacked_ransac);
        cvDestroyWindow(IMG_MATCH2);

        //释放全景拼接图像
        if(xformed)
        {
            cvReleaseImage(&xformed);
            cvReleaseImage(&xformed_simple);
            cvReleaseImage(&xformed_proc);
            cvDestroyWindow(IMG_MOSAIC_TEMP);
            cvDestroyWindow(IMG_MOSAIC_SIMPLE);
            cvDestroyWindow(IMG_MOSAIC_BEFORE_FUSION);
            cvDestroyWindow(IMG_MOSAIC_PROC);
        }
    }
    verticalStackFlag = false;//显示匹配结果的合成图片的排列方向标识复位
	
}


