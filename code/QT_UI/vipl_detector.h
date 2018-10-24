#ifndef VIPL_DETECTOR_H
#define VIPL_DETECTOR_H

#include <QMainWindow>

namespace Ui {
class VIPL_Detector;
}

class VIPL_Detector : public QMainWindow
{
    Q_OBJECT

public:
    explicit VIPL_Detector(QWidget *parent = 0);
	void resetGrid(QWidget * widget, double factorx, double factory);
    ~VIPL_Detector();

private slots:
    //int on_pushButton_clicked();
	void on_ButtonImage_1_clicked();
	void on_ButtonImage_2_clicked();
	void on_ButtonTXT_clicked();
	void on_ButtonStop_clicked();

	int on_ButtonReal_clicked();

	void on_ButtonVedio_clicked();
	void on_ButtonMulti_clicked();

	

private:
    Ui::VIPL_Detector *ui;
};

#endif // VIPL_DETECTOR_H
