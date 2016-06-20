#include "easypr/chars_recognise.h"
#include "easypr/util.h"

/*! \namespace easypr
Namespace where all the C++ EasyPR functionality resides
*/
namespace easypr {

CCharsRecognise::CCharsRecognise() {
  m_charsSegment = new CCharsSegment();
  m_charsIdentify = new CCharsIdentify();
}

CCharsRecognise::~CCharsRecognise() {
  SAFE_RELEASE(m_charsSegment);
  SAFE_RELEASE(m_charsIdentify);
}

void CCharsRecognise::LoadANN(string s) {
  m_charsIdentify->LoadModel(s.c_str());
}

string CCharsRecognise::charsRecognise(Mat plate) {
  return m_charsIdentify->charsIdentify(plate);
}
int CCharsRecognise::charsRecognise(Mat plate, string& plateLicense) {
  //车牌字符方块集合
  vector<Mat> matVec;

  string plateIdentify = "";

  int result = m_charsSegment->charsSegment(plate, matVec);
  int string_length = 0;
  if (result == 0) {
    int num = matVec.size();
    for (int j = 0; j < num; j++) {
      Mat charMat = matVec[j];
      
       stringstream ss;
       ss<<j;
       string s1 = ss.str() + ".jpg";
       s1 = "tmp_character/" + s1;
       imwrite(s1.c_str(), charMat);
      
      bool isChinses = false;

      bool isSpeci = false;

      //默认首个字符块是中文字符
      if (j == 0) isChinses = true;
      if (j == 1) isSpeci = true;

      string charcater =
          m_charsIdentify->charsIdentify(charMat, isChinses, isSpeci);

      plateIdentify = plateIdentify + charcater;
      string_length ++;
      // std::cout << plateIdentify << std::endl;
      //std::cout << " size:  " << plateIdentify.size() << std::endl;
    }
  }

  plateLicense = plateIdentify;
  //std::cout << plateLicense << std::endl;
  //std::cout << plateLicense.size() << std::endl;
  
  
  if (plateLicense.size() < 7) {
    return -1;
  }

  return result;
}

} /*! \namespace easypr*/
