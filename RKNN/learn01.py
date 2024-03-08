from rknn.api import RKNN

if __name__ == '__main__':

    # 创建RKNN对象
    rknn = RKNN(verbose=True)

    # 设置模型转换参数，这里可以指定平台，添加target_platform='rk3588'配置，默认rk3566
    # mean_values是设置输入的均值，std_values是输入的归一化值
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])
    print('done')

    # 导入onnx模型，使用model指定onnx模型路径
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # 构建RKNN模型，这里设置do_quantization为true开启量化，dataset是指定用于量化校正的数据集
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # 导出RKNN模型，使用export_path指定导出模型路径，这里默认设置RKNN_MODEL
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # 调用init_runtime接口初始化运行时环境，默认是在PC上模拟仿真
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # 设置输出，用于模型推理
    img = cv2.imread(IMG_PATH)
    # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 进行推理，没有设置target默认使用模拟器，之后对输出数据后处理并保存结果
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    np.save('./onnx_yolov5_0.npy', outputs[0])
    np.save('./onnx_yolov5_1.npy', outputs[1])
    np.save('./onnx_yolov5_2.npy', outputs[2])
    print('done')

    # 省略...
