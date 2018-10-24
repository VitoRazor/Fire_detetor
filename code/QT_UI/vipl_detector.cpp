#include "vipl_detector.h"
#include "ui_vipl_detector.h"
#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include<QFileDialog>

#ifdef _WIN32
#define OPENCV
#endif

#include "VIPL_class.hpp"	// imported functions from DLL

//#ifdef OPENCV
#include <opencv2/opencv.hpp>			// C++
#include "opencv2/core/version.hpp"
#include "opencv2/videoio/videoio.hpp"
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp> 
#include <fstream>  
#include <string>  
#include <iostream> 
#include <QDesktopWidget>

//#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)""CVAUX_STR(CV_VERSION_REVISION)
//#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
//#else
//#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)""CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)
//#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
//#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
//#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
//#endif


using namespace cv;
using namespace std;
bool stop = false;

struct multidata {
	String filenmae;
	Mat img;
};
extern string line_1, line_2, line_3;
Detector detector;
vector<string> obj_names;
vector<String> vecimg1;
vector<String> vecvideo;
queue<multidata> Qdata;
int multithread = 2;
std::mutex mtx1;
std::mutex mtx2;

/*#define CV_8U   0
#define CV_8S   1
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6*/


QImage cvMat_to_QImage(const cv::Mat &mat) {
	switch (mat.type())
	{
		// 8-bit, 4 channel
	case CV_8UC4:
	{
		QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB32);
		return image;
	}

	// 8-bit, 3 channel
	case CV_8UC3:
	{
		QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
		return image.rgbSwapped();
	}

	// 8-bit, 1 channel
	case CV_8UC1:
	{
		static QVector<QRgb>  sColorTable;
		// only create our color table once
		if (sColorTable.isEmpty())
		{
			for (int i = 0; i < 256; ++i)
				sColorTable.push_back(qRgb(i, i, i));
		}
		QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
		image.setColorTable(sColorTable);
		return image;
	}

	default:
		//qDebug("Image format is not supported: depth=%d and %d channels\n", mat.depth(), mat.channels());
		break;
	}
	return QImage();
}
cv::Mat QImage2cvMat(QImage image)
{
	cv::Mat mat;
	//qDebug() << image.format();
	switch (image.format())
	{
	case QImage::Format_ARGB32:
	case QImage::Format_RGB32:
	case QImage::Format_ARGB32_Premultiplied:
		mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
		break;
	case QImage::Format_RGB888:
		mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
		cv::cvtColor(mat, mat, CV_BGR2RGB);
		break;
	case QImage::Format_Indexed8:
		mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
		break;
	}
	return mat;
}
IplImage *QImageToIplImage(QImage qImage)
{
	int width = qImage.width();
	int height = qImage.height();
	CvSize Size;
	Size.height = height;
	Size.width = width;
	IplImage *IplImageBuffer = cvCreateImage(Size, IPL_DEPTH_8U, 3);
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			QRgb rgb = qImage.pixel(x, y);
			cvSet2D(IplImageBuffer, y, x, CV_RGB(qRed(rgb), qGreen(rgb), qBlue(rgb)));
		}
	}
	return IplImageBuffer;
}

class track_kalman {
public:
	cv::KalmanFilter kf;
	int state_size, meas_size, contr_size;


	track_kalman(int _state_size = 10, int _meas_size = 10, int _contr_size = 0)
		: state_size(_state_size), meas_size(_meas_size), contr_size(_contr_size)
	{
		kf.init(state_size, meas_size, contr_size, CV_32F);

		cv::setIdentity(kf.measurementMatrix);
		cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
		cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-5));
		cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1e-2));
		cv::setIdentity(kf.transitionMatrix);
	}

	void set(std::vector<bbox_t> result_vec) {
		for (size_t i = 0; i < result_vec.size() && i < state_size * 2; ++i) {
			kf.statePost.at<float>(i * 2 + 0) = result_vec[i].x;
			kf.statePost.at<float>(i * 2 + 1) = result_vec[i].y;
		}
	}

	// Kalman.correct() calculates: statePost = statePre + gain * (z(k)-measurementMatrix*statePre);
	// corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
	std::vector<bbox_t> correct(std::vector<bbox_t> result_vec) {
		cv::Mat measurement(meas_size, 1, CV_32F);
		for (size_t i = 0; i < result_vec.size() && i < meas_size * 2; ++i) {
			measurement.at<float>(i * 2 + 0) = result_vec[i].x;
			measurement.at<float>(i * 2 + 1) = result_vec[i].y;
		}
		cv::Mat estimated = kf.correct(measurement);
		for (size_t i = 0; i < result_vec.size() && i < meas_size * 2; ++i) {
			result_vec[i].x = estimated.at<float>(i * 2 + 0);
			result_vec[i].y = estimated.at<float>(i * 2 + 1);
		}
		return result_vec;
	}

	// Kalman.predict() calculates: statePre = TransitionMatrix * statePost;
	// predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
	std::vector<bbox_t> predict() {
		std::vector<bbox_t> result_vec;
		cv::Mat control;
		cv::Mat prediction = kf.predict(control);
		for (size_t i = 0; i < prediction.rows && i < state_size * 2; ++i) {
			result_vec[i].x = prediction.at<float>(i * 2 + 0);
			result_vec[i].y = prediction.at<float>(i * 2 + 1);
		}
		return result_vec;
	}

};

class extrapolate_coords_t {
public:
	std::vector<bbox_t> old_result_vec;
	std::vector<float> dx_vec, dy_vec, time_vec;
	std::vector<float> old_dx_vec, old_dy_vec;

	void new_result(std::vector<bbox_t> new_result_vec, float new_time) {
		old_dx_vec = dx_vec;
		old_dy_vec = dy_vec;
		if (old_dx_vec.size() != old_result_vec.size()) //std::cout << "old_dx != old_res \n";
		dx_vec = std::vector<float>(new_result_vec.size(), 0);
		dy_vec = std::vector<float>(new_result_vec.size(), 0);
		update_result(new_result_vec, new_time, false);
		old_result_vec = new_result_vec;
		time_vec = std::vector<float>(new_result_vec.size(), new_time);
	}

	void update_result(std::vector<bbox_t> new_result_vec, float new_time, bool update = true) {
		for (size_t i = 0; i < new_result_vec.size(); ++i) {
			for (size_t k = 0; k < old_result_vec.size(); ++k) {
				if (old_result_vec[k].track_id == new_result_vec[i].track_id && old_result_vec[k].obj_id == new_result_vec[i].obj_id) {
					float const delta_time = new_time - time_vec[k];
					if (abs(delta_time) < 1) break;
					size_t index = (update) ? k : i;
					float dx = ((float)new_result_vec[i].x - (float)old_result_vec[k].x) / delta_time;
					float dy = ((float)new_result_vec[i].y - (float)old_result_vec[k].y) / delta_time;
					float old_dx = dx, old_dy = dy;

					// if it's shaking
					if (update) {
						if (dx * dx_vec[i] < 0) dx = dx / 2;
						if (dy * dy_vec[i] < 0) dy = dy / 2;
					}
					else {
						if (dx * old_dx_vec[k] < 0) dx = dx / 2;
						if (dy * old_dy_vec[k] < 0) dy = dy / 2;
					}
					dx_vec[index] = dx;
					dy_vec[index] = dy;

					//if (old_dx == dx && old_dy == dy) std::cout << "not shakin \n";
					//else std::cout << "shakin \n";

					if (dx_vec[index] > 1000 || dy_vec[index] > 1000) {
						//std::cout << "!!! bad dx or dy, dx = " << dx_vec[index] << ", dy = " << dy_vec[index] << 
						//	", delta_time = " << delta_time << ", update = " << update << std::endl;
						dx_vec[index] = 0;
						dy_vec[index] = 0;
					}
					old_result_vec[k].x = new_result_vec[i].x;
					old_result_vec[k].y = new_result_vec[i].y;
					time_vec[k] = new_time;
					break;
				}
			}
		}
	}

	std::vector<bbox_t> predict(float cur_time) {
		std::vector<bbox_t> result_vec = old_result_vec;
		for (size_t i = 0; i < old_result_vec.size(); ++i) {
			float const delta_time = cur_time - time_vec[i];
			auto &bbox = result_vec[i];
			float new_x = (float)bbox.x + dx_vec[i] * delta_time;
			float new_y = (float)bbox.y + dy_vec[i] * delta_time;
			if (new_x > 0) bbox.x = new_x;
			else bbox.x = 0;
			if (new_y > 0) bbox.y = new_y;
			else bbox.y = 0;
		}
		return result_vec;
	}

};

cv::Mat draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
	int current_det_fps = -1, int current_cap_fps = -1)
{
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

	for (auto &i : result_vec) {
		cv::Scalar color = obj_id_to_color(i.obj_id);
		
		if (obj_names.size() > i.obj_id) {

			std::string obj_name = obj_names[i.obj_id];
			//if (obj_name == "car" || obj_name == "pedestrian" || obj_name == "cyclist") {
			//	cv::Scalar color1;
			//	if (obj_name == "car") {
			//		color1 = cv::Scalar(0, 0, 255);
			//	}
			//	else if (obj_name == "pedestrian") {
			//		color1 = cv::Scalar(255, 0, 0);
			//	}
			//	else if (obj_name == "cyclist") {
			//		color1 = cv::Scalar(0, 255, 0);

			//	}
				cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
				if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
				cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
				//int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
				int const max_width = text_size.width;
				cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 22, 0)),
					cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
					color, CV_FILLED, 8, 0);
				putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 8), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			}
	}
	if(current_det_fps >= 0 && current_cap_fps >= 0) {
		std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
		putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(50, 200, 0), 2);
	}
	return mat_img;
}




QString show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names);
void capvideo(cv::VideoCapture cap, String name) {

	Mat frame;
	//
	while (!stop)
	{
		cap >> frame;
		if (frame.empty()) {
			break;
		}
		//multidata x;
		//x.filenmae = name;
		//x.img = frame;
		//std::unique_lock<std::mutex> lock(mtx1);
		mtx1.lock();
		//Qdata.push(x);
		
		std::vector<bbox_t> result_vec = detector.detect(frame);
		QString result = show_console_result(result_vec, obj_names);
		cout<< result.toStdString()<<endl;
		mtx1.unlock();
		draw_boxes(frame, result_vec, obj_names);
		imshow(name, frame);
		if (waitKey(30) == 27)
			stop = true;
		if (stop) {
			destroyAllWindows();
		}
		
		//if (stop) {
		//	destroyAllWindows();
		//}
	}
}

void multidetection(String filename, String windowsname) {

}




QString show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
	QString  displayer = "";
	for (auto &i : result_vec) {
		QString str = "";
		if (obj_names.size() > i.obj_id) {
			
			//std::cout << obj_names[i.obj_id] << " - ";
			str.sprintf("%s %s", obj_names[i.obj_id], " - ");
			displayer += str;
			//ui->display->append(str);
			str.clear();
			
		}
		/*std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
			<< ", w = " << i.w << ", h = " << i.h
			<< std::setprecision(3) << ", prob = " << i.prob << std::endl;
			*/
		str.sprintf("obj_id=%d,  x=%d, y=%d, w=%d, h=%d, prob=%0.2f\n", i.obj_id, i.x, i.y, i.w, i.h, i.prob);
		displayer += str;
		str.clear();
	}
	displayer += "=======================================";
	return displayer;
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for (std::string line; getline(file, line);) file_lines.push_back(line);
	//std::cout << "object names loaded \n";
	return file_lines;
}

VIPL_Detector::VIPL_Detector(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::VIPL_Detector)
{
    ui->setupUi(this);
	detector.Detector_init(line_1, line_2);
	obj_names= objects_names_from_file(line_3);
	std::cout << line_1 << std::endl;
	std::cout << line_2 << std::endl;
	std::cout << line_3 << std::endl;
	QDesktopWidget *deskWgt = QApplication::desktop();
	int currentScreenWid = deskWgt->width();
	int currentScreenHei = deskWgt->height();


	//resetGrid(this, currentScreenWid / 1920.0, currentScreenHei / 1080.0);
	//resetGrid(ui->verticalLayout, currentScreenWid / 1920.0, currentScreenHei / 1080.0);
	//resetGrid(ui->verticalLayout, currentScreenWid / 1920.0, currentScreenHei / 1080.0);
	//resetGrid(ui->verticalLayout, currentScreenWid / 1920.0, currentScreenHei / 1080.0);


	//QPixmap *pixmap = new QPixmap("sculogo.bmp");
	QPixmap *pixmap = new QPixmap("fire.jpg");
	pixmap->scaled(ui->label->size(), Qt::KeepAspectRatio);
	ui->label->setScaledContents(true);
	ui->label->setPixmap(*pixmap);
}

VIPL_Detector::~VIPL_Detector()
{
	
    delete ui;
}

void VIPL_Detector::on_ButtonImage_1_clicked()
{
	
	QString filename = QFileDialog::getOpenFileName(this,
		tr("Select Image"),
		"",
		tr("Images (*.png *.bmp *.jpg *.tif *.GIF)")); //选择路径  
	if (filename.isEmpty())
	{
		return;
	}
	else
	{
		cv::Mat mat_img = cv::imread(filename.toStdString());

		auto start = std::chrono::steady_clock::now();
		std::vector<bbox_t> result_vec = detector.detect(mat_img);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> spent = end - start;
		std::cout << " Time: " << spent.count() << " sec \n";

		//result_vec = detector.tracking_id(result_vec);	// comment it - if track_id is not required
		Mat img2 = cv::imread("depth/1.png");
		draw_boxes(mat_img, result_vec, obj_names);
		cv::imshow("window name", mat_img);
		draw_boxes(img2, result_vec, obj_names);
		cv::imwrite("depth/2_predict.jpg", img2);
		ui->display->append(show_console_result(result_vec, obj_names));
		cv::waitKey(0);
		cv::imwrite("predect.jpg", mat_img);
	}
}

void VIPL_Detector::on_ButtonImage_2_clicked()
{
	//auto obj_names = objects_names_from_file(line_3);
	QString folderName = QFileDialog::getExistingDirectory(this, tr("Open Folder"), QString());
	if (!folderName.isEmpty()) {
		QDir dir(folderName);
		vecimg1.clear();
		foreach(QFileInfo imageFileInfo, dir.entryInfoList(QStringList() << "*.jpg" << "*.png" << "*.bmp", QDir::Files, QDir::NoSort))
		{
			QString  img1Name = imageFileInfo.absoluteFilePath();
			//QImage img1(img1Name);
			vecimg1.push_back(img1Name.toStdString());
		}
	}
	if (!vecimg1.empty()) {
		vector<String>::iterator it = vecimg1.begin();
		while (!stop&&it != vecimg1.end()) {

			cv::Mat mat_img = cv::imread(*it);

			auto start = std::chrono::steady_clock::now();
			std::vector<bbox_t> result_vec = detector.detect(mat_img);
			auto end = std::chrono::steady_clock::now();
			std::chrono::duration<double> spent = end - start;
			std::cout << " Time: " << spent.count() << " sec \n";

			//result_vec = detector.tracking_id(result_vec);	// comment it - if track_id is not required
			draw_boxes(mat_img, result_vec, obj_names);
			cv::imshow("Image", mat_img);
			ui->display->append(show_console_result(result_vec, obj_names));
			cv::waitKey(0);
			it++;
		}
		
	}
}
void VIPL_Detector::on_ButtonTXT_clicked()
{
	
	QString filename = QFileDialog::getOpenFileName(this,
		tr("Select TXT"),
		"",
		tr("TXT (*.txt)")); //选择路径  
	if (filename.isEmpty())
	{
		return;
	}
	else{
		std::ifstream file(filename.toStdString());
		if (!file.is_open()) { std::cout << "File not found! \n";
		return;
		}
		else
			for (std::string line; file >> line;) {
				//std::cout << line << std::endl;
				cv::Mat mat_img = cv::imread(line);
				std::vector<bbox_t> result_vec = detector.detect(mat_img);
				//show_console_result(result_vec, obj_names);
				ui->display->append(show_console_result(result_vec, obj_names));
	}
			}
	
}

void VIPL_Detector::on_ButtonStop_clicked()
{
	stop = true;
}

int VIPL_Detector::on_ButtonReal_clicked()
{
	stop = false;
	std::string filename;
	auto obj_names = objects_names_from_file(line_3);
	std::string out_videofile = "result.avi";
	bool const save_output_videofile = false;
	int mode = 2;
	float thread_private = 0.16;

	if (mode == 2) {
		//std::cout << "请选择摄像头（0，1，3.....）：\n";
		int capture_flage = 1;
		//std::cin >> capture_flage;
		VideoCapture cap(capture_flage);
		if (!cap.isOpened())
		{
			return -1;
		}
		//Mat frame;
		Mat mat_img;
		//std::cout << "按Esc退出\n";
		while (!stop)
		{
			try {
#ifdef OPENCV
				cap >> mat_img;
				std::vector<bbox_t> result_vec = detector.detect(mat_img);
				draw_boxes(mat_img, result_vec, obj_names);
				//show_console_result(result_vec, obj_names);
				//QImage bbb = cvMat_to_QImage(aaa);
				//this->ui->label->setPixmap(QPixmap::fromImage(bbb));
				//this->ui->label->update();
				//repaint();//非常重要  刷新界面
				cv::imshow("real-time Detector", mat_img);
				//update();
				//cv::imshow("real-time Detector", mat_img);
				if (waitKey(30)==27)
					stop = true;
				if (stop) {
					destroyAllWindows();
				}
				
				//show_console_result(result_vec, obj_names);
				//}
#else
				//std::vector<bbox_t> result_vec = detector.detect(filename);

				auto img = detector.load_image(filename);
				std::vector<bbox_t> result_vec = detector.detect(img);
				detector.free_image(img);
				show_console_result(result_vec, obj_names);
#endif			
			}
			catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
			catch (...) { std::cerr << "unknown exception \n"; getchar(); }
			filename.clear();
		}
	}
	return 0;
}
void VIPL_Detector::on_ButtonVedio_clicked() {
	
	stop = false;
	QString filename1 = QFileDialog::getOpenFileName(this,
		tr("Slelect Vedio"),
		"",
		tr("vedio (*.avi *.mp4 *.mjpg *.mov)")); //选择路径
	std::string out_videofile = "result.avi";
	bool const save_output_videofile = true;
	if (filename1.isEmpty())
	{
		return;
	}
	else
	{
		String filename=filename1.toStdString();
		try {
			std::string const file_ext = filename.substr(filename.find_last_of(".") + 1);
			std::string const protocol = filename.substr(0, 7);
			if (file_ext == "avi" || file_ext == "mp4" || file_ext == "mjpg" || file_ext == "mov" || 	// video file
				protocol == "rtmp://" || protocol == "rtsp://" || protocol == "http://" || protocol == "https:/")	// video network stream
			{
				cv::Mat cap_frame, cur_frame, det_frame, write_frame;
				std::shared_ptr<image_t> det_image;
				std::vector<bbox_t> result_vec, thread_result_vec;
				detector.nms = 0.02;	// comment it - if track_id is not required
				std::atomic<bool> consumed, videowrite_ready;
				consumed = true;
				videowrite_ready = true;
				std::atomic<int> fps_det_counter, fps_cap_counter;
				fps_det_counter = 0;
				fps_cap_counter = 0;
				int current_det_fps = 0, current_cap_fps = 0;
				std::thread t_detect, t_cap, t_videowrite;
				std::mutex mtx;
				std::condition_variable cv;
				std::chrono::steady_clock::time_point steady_start, steady_end;
				cv::VideoCapture cap(filename);
				cap >> cur_frame;
				int const video_fps = cap.get(CV_CAP_PROP_FPS);
				cv::Size const frame_size = cur_frame.size();
				cv::VideoWriter output_video;
				if (save_output_videofile)
					output_video.open(out_videofile, CV_FOURCC('D', 'I', 'V', 'X'), std::max(35, video_fps), frame_size, true);
				int k = 0;
				while (!cur_frame.empty()) {
					if (t_cap.joinable()) {
						t_cap.join();
						++fps_cap_counter;
						cur_frame = cap_frame.clone();
					}
					t_cap = std::thread([&]() {
						if(!stop)
							cap >> cap_frame;
						else 
							cap_frame.release();
					});

					// swap result and input-frame
					if (consumed)
					{
						std::unique_lock<std::mutex> lock(mtx);
						det_image = detector.mat_to_image_resize(cur_frame);
						result_vec = thread_result_vec;
						//std::vector<bbox_t> result_vec = detector.detect(mat_img);
						//result_vec = detector.tracking_id(result_vec);	// comment it - if track_id is not required
						consumed = false;
					}
					// launch thread once
					if (!t_detect.joinable()) {
						t_detect = std::thread([&]() {
							auto current_image = det_image;
							consumed = true;
							while (current_image.use_count() > 0) {									
								auto result = detector.detect_resized(*current_image, frame_size.width, frame_size.height, 0.35, true);
								++fps_det_counter;
								std::unique_lock<std::mutex> lock(mtx);
								thread_result_vec = result;
								current_image = det_image;
								consumed = true;
								cv.notify_all();
							}
						});
					}

					if (!cur_frame.empty()) {
						steady_end = std::chrono::steady_clock::now();
						if (std::chrono::duration<double>(steady_end - steady_start).count() >= 1) {
							current_det_fps = fps_det_counter;
							current_cap_fps = fps_cap_counter;
							steady_start = steady_end;
							fps_det_counter = 0;
							fps_cap_counter = 0;
						}
						draw_boxes(cur_frame, result_vec, obj_names, current_det_fps, current_cap_fps);
						ui->display->append(show_console_result(result_vec, obj_names));
						cv::imshow("vedio-detector", cur_frame);
						int key = cv::waitKey(3);
						if (output_video.isOpened() && videowrite_ready) {
							if (t_videowrite.joinable()) t_videowrite.join();
							write_frame = cur_frame.clone();
							videowrite_ready = false;
							t_videowrite = std::thread([&]() {
								output_video << write_frame; videowrite_ready = true;
							});
						}
					}

					// wait detection result for video-file only (not for net-cam)
					if (protocol != "rtsp://" && protocol != "http://" && protocol != "https:/") {
						std::unique_lock<std::mutex> lock(mtx);
						while (!consumed) cv.wait(lock);
					}
				}
				if (t_cap.joinable()) t_cap.join();
				if (t_detect.joinable()) t_detect.join();
				if (t_videowrite.joinable()) t_videowrite.join();
				std::cout << "Video ended \n";
				cvDestroyAllWindows();
			}
		}
		catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
		catch (...) { std::cerr << "unknown exception \n"; getchar(); }
		filename.clear();
	}
	return;
}

void VIPL_Detector::on_ButtonMulti_clicked()
{
	QString folderName = QFileDialog::getExistingDirectory(this, tr("Open Folder"), QString());
	if (!folderName.isEmpty()) {
		QDir dir(folderName);
		vecvideo.clear();
		foreach(QFileInfo imageFileInfo, dir.entryInfoList(QStringList() << "*.avi" << "*.mp4" << "*.mov", QDir::Files, QDir::NoSort))
		{
			QString  img1Name = imageFileInfo.absoluteFilePath();
			//QImage img1(img1Name);
			vecvideo.push_back(img1Name.toStdString());
		}
	}
	if (!vecvideo.empty()) {
		vector<String>::iterator it = vecvideo.begin();
		VideoCapture cap[10];
		cv::Mat cap_frame[10], cur_frame[10];
		while (!stop && it != vecvideo.end()) {
			int i ;
			std::thread cap_thread[10];
			std::thread det_thread[10];



			for (i = 0; i < multithread; i++) {
				if (it + i == vecvideo.end()) {
					break;
				}
				//begin to capture image
				char sss[20];
				sprintf(sss, "video%d", i);
				cap[i].open(*(it + i));
				cap_thread[i] = std::thread(capvideo,cap[i], sss);				
			}
			cout << i << endl;
			for (int j = 0; j < i; j++) {
				cap_thread[j].join();
			}
			//std::unique_lock<std::mutex> lock(mtx1);
/*			while (!Qdata.empty()) {
				mtx1.lock();
				multidata e = Qdata.front();
				cout << e.filenmae << endl;
				mtx2.lock();
				Mat mat_img = e.img.clone();
				Qdata.pop();
				mtx1.unlock();
				std::vector<bbox_t> result_vec = detector.detect(mat_img);
				draw_boxes(mat_img, result_vec, obj_names);
				imshow("11", mat_img);
				waitKey(3);
				mtx2.unlock();
				
			}*/
			
			it = it + i;
			
		}

	}
}

void VIPL_Detector::resetGrid(QWidget *widget, double factorx, double factory)
{
	int widgetX = widget->x();
	int widgetY = widget->y();
	int widgetWid = widget->width();
	int widgetHei = widget->height();
	int nWidgetX = (int)(widgetX*factorx);
	int nWidgetY = (int)(widgetY*factory);
	int nWidgetWid = (int)(widgetWid*factorx);
	int nWidgetHei = (int)(widgetHei*factory);
	widget->setGeometry(nWidgetX, nWidgetY, nWidgetWid, nWidgetHei);
}