LOCAL_PATH := $(call my-dir) 

include $(CLEAR_VARS)

OPENCV_CAMERA_MODULES:=on
OPENCV_INSTALL_MODULES:=on
 
include ../OpenCV-2.4.9-android-sdk/sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := siftConjunction
LOCAL_SRC_FILES := sift_conjunction.cpp \
imgfeatures.c \
kdtree.c \
minpq.c \
sift.c \
utils.c \
xform.c \

LOCAL_C_INCLUDES := imgfeatures.h \
kdtree.h \
minpq.h \
sift.h \
siftmatch.h \
utils.h \
xform.h \

LOCAL_LDLIBS     += -llog -ldl

include $(BUILD_SHARED_LIBRARY)
