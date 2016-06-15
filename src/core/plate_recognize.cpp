#include "easypr/plate_recognize.h"

/*! \namespace easypr
    Namespace where all the C++ EasyPR functionality resides
*/
namespace easypr {

CPlateRecognize::CPlateRecognize() {
  // cout << "CPlateRecognize" << endl;
  // m_plateDetect= new CPlateDetect();
  // m_charsRecognise = new CCharsRecognise();
}

// !����ʶ��ģ��
int CPlateRecognize::plateRecognize(Mat src, std::vector<string> &licenseVec) {
  // ���Ʒ��鼯��
  vector<CPlate> plateVec;
  vector<CPlate> truePlateVec;

  // ������ȶ�λ��ʹ����ɫ��Ϣ�����Sobel
  int resultPD = plateDetect(src, plateVec, getPDDebug(), 0);

  if (resultPD == 0) {
    int num = plateVec.size();
    int index = 0;

    //����ʶ��ÿ�������ڵķ���
    for (int j = 0; j < num; j++) {
      CPlate item = plateVec[j];
      Mat plate = item.getPlateMat();

      //��ȡ������ɫ
      string plateType = getPlateColor(plate);

      //��ȡ���ƺ�
      string plateIdentify = "";
      int resultCR = charsRecognise(plate, plateIdentify);
      vector<int> plate_pos(4,0);
      if (resultCR == 0) {
        RotatedRect rec_roi = item.getPlatePos();
        double angle = rec_roi.angle;
        Point pt = rec_roi.center;
        Size platesize = rec_roi.size;
        if (std::fabs(rec_roi.angle) < 45) {
          plate_pos[0] = pt.x - platesize.width / 2;
          plate_pos[1] = pt.x + platesize.width / 2;
          plate_pos[2] = pt.y - platesize.height / 2;
          plate_pos[3] = pt.y + platesize.height / 2;
        }
        else {
          plate_pos[0] = pt.x - platesize.height / 2;
          plate_pos[1] = pt.x + platesize.height / 2;
          plate_pos[2] = pt.y - platesize.width / 2;
          plate_pos[3] = pt.y + platesize.width / 2;          
        }
        string license = plateIdentify;
        stringstream ss;
        for (int _i=0; _i < plate_pos.size(); _i++) {
          ss << plate_pos[_i];
          license = license + " " + ss.str();
          ss.str("");
        }
        
        //string license = plateType + ":" + plateIdentify;
        licenseVec.push_back(license);
        truePlateVec.push_back(item);
        //RotatedRect rec_roi = item.getPlatePos();
        //double angle = rec_roi.angle;
        //Point pt = rec_roi.center;
        //Size platesize = rec_roi.size;
        //std::cout << angle << " " << platesize.width << "  "  << platesize.height << std::endl;
      }
    }
    //����ʶ����̵��˽���

    //�����Debugģʽ������Ҫ����λ��ͼƬ��ʾ��ԭͼ���Ͻ�
    if (getPDDebug() == true) {
      Mat result;
      src.copyTo(result);
      int truenum = truePlateVec.size();

      for (int j = 0; j < truenum; j++) {
        CPlate item = truePlateVec[j];
        Mat plate = item.getPlateMat();

        int height = 36;
        int width = 136;
        if (height * index + height < result.rows) {
          Mat imageRoi = result(Rect(0, 0 + height * index, width, height));
          addWeighted(imageRoi, 0, plate, 1, 0, imageRoi);
        }
        index++;

        RotatedRect minRect = item.getPlatePos();
        Point2f rect_points[4];
        minRect.points(rect_points);


        Scalar lineColor = Scalar(255, 255, 255);

        if (item.getPlateLocateType() == SOBEL) lineColor = Scalar(255, 0, 0);

        if (item.getPlateLocateType() == COLOR) lineColor = Scalar(0, 255, 0);

        for (int j = 0; j < 4; j++)
          line(result, rect_points[j], rect_points[(j + 1) % 4], lineColor, 2,
               8);
      }

      //��ʾ��λ���ͼƬ
      //showResult(result);
      //imwrite("result.jpg", result);
    }
  }

  return resultPD;
}

} /*! \namespace easypr*/
