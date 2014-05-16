# The ARMv7 is significanly faster due to the use of the hardware FPU
APP_ABI := armeabi-v7a armeabi x86
APP_PLATFORM := android-8
APP_STL := gnustl_static
APP_CPPFLAGS := -frtti -fexceptions
