import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if sys.platform.startswith("win"):
    sys.path.append(str(ROOT / "MvImport"))
    from MvImport.MvCameraControl_class import MvCamera
    from MvImport.CameraParams_header import MV_GIGE_DEVICE, MV_USB_DEVICE, MV_CC_DEVICE_INFO_LIST
else:
    sys.path.append(str(ROOT / "MvImport_Linux"))
    from MvImport_Linux.MvCameraControl_class import MvCamera
    from MvImport_Linux.CameraParams_header import MV_GIGE_DEVICE, MV_USB_DEVICE, MV_CC_DEVICE_INFO_LIST
from ctypes import byref, cast, POINTER

deviceList = MV_CC_DEVICE_INFO_LIST()
ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
print("Enum ret:", ret, "device count:", deviceList.nDeviceNum)

if deviceList.nDeviceNum == 0:
    raise SystemExit("No camera detected by SDK.")
print("Camera detected")
