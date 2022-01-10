#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <highgui.hpp>
#include <imgproc.hpp>
#include <core.hpp>
#include <ctime>
#include <cuda_runtime_api.h>

//读取单张图片，转换成灰度图，调整分辨率
//参数path: 图片路径
//返回：Mat类型图像
cv::Mat readImage(std::string path) {
	cv::Mat image = cv::imread(path);
	cv::resize(image, image, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
	return image;
}

//读取多张图片，并竖向拼接
//参数paths: 图片路径
//返回：Mat类型图像
cv::Mat readImages(std::vector<const char*> paths) {
	int batch = paths.size();
	cv::Mat images = readImage(paths[0]);
	cv::Mat image;
	for (int i = 1; i < batch; i++) {
		image = readImage(paths[i]);
		cv::vconcat(images, image, images);
	}
	return images;
}

//Mat图像转Tensor，并将0—255的像素值归一化到0—1
//参数input: 输入图像
//参数c_number:输入图像通道数
//参数b_size:输入图像数量
//返回：四维Tensor
torch::Tensor mat2tensor(cv::Mat input, int c_number, int b_size) {
	torch::Tensor tensor_image = torch::from_blob(input.data, { b_size, input.rows/b_size, input.cols, c_number }, torch::kByte);
	tensor_image = tensor_image.permute({ 0,3,1,2 });
	tensor_image = tensor_image.toType(torch::kFloat);
	tensor_image = tensor_image.div(255);
	return tensor_image;
}

torch::jit::script::Module load_model(const char* model_path, int cuda_id = 0) {
	//std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
	//std::cout << "cuda::cudnn_is_available():" << torch::cuda::cudnn_is_available() << std::endl;
	//std::cout << "cuda::device_count():" << torch::cuda::device_count() << std::endl;

	if (cuda_id >= 0) {
		cudaSetDevice(cuda_id);
	}
	torch::NoGradGuard no_grad_guard;
	torch::jit::script::Module module = torch::jit::load(model_path);	//读取模型参数	

	if (cuda_id >= 0) {
		module.to(at::kCUDA);
	}

	module.eval();
	return module;
}

int main() {
	std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
	std::cout << "cuda::cudnn_is_available():" << torch::cuda::cudnn_is_available() << std::endl;
	std::cout << "cuda::device_count():" << torch::cuda::device_count() << std::endl;
	
	//cudaSetDevice(0);
	clock_t start, end;
	torch::NoGradGuard no_grad_guard;

	// Deserialize the ScriptModule from a file using torch::jit::load().
	torch::jit::script::Module module = load_model("best.torchscript.pt", -1);	//读取模型参数	

	// Create a vector of inputs.
	//std::vector<torch::jit::IValue> inputs;	//声明模型输入
	std::string path;

	//图片路径
	std::ifstream val_list("image_list_val.txt");

	double time_sum = 0;
	for (int i = 0; i < 3226; i++) {
		val_list >> path;
		cv::Mat images = readImage("D:/WIDER_FACE/WIDER_val/images/"+path);

		torch::Tensor tensor_image;
		tensor_image = mat2tensor(images, 3, 1);// .to(at::kCUDA);

		//inputs.push_back(tensor_image);

		start = clock();		//程序开始计时
		torch::Tensor output = module.forward({ tensor_image }).toTensor();	//识别预测
		end = clock();		//程序结束用时
		time_sum = time_sum+(double)(end - start);
		//std::cout << "Total time:" << endtime / CLOCKS_PER_SEC<< std::endl;		//s为单位
	}
	std::cout << "Total time:" << time_sum << "ms" << std::endl;	//ms为单位
	std::cout << "FPS:" << 3226 * 1000 / time_sum << "ms" << std::endl;	//ms为单位
	/*
	torch::Tensor result = output.argmax(1, true);	//获取识别结果

	int n = result.sizes()[0];
	int predict;
	for (int i = 0; i < n; i++) {
		predict = result[i].item().toInt();
		std::cout << predict << std::endl;
	}
	*/
}