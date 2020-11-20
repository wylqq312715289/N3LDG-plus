#ifndef _MYLIB_H_
#include <fstream>
#define _MYLIB_H_

#include <functional>
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <deque>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <json/json.h>

#include "NRMat.h"
#include "Eigen/Dense"
#include "Def.h"

using namespace nr;
using namespace std;
using namespace Eigen;

typedef long long blong;

const static dtype minlogvalue = -1000;
const static dtype d_zero = 0.0;
const static dtype d_one = 1.0;
const static string nullkey = "-NULL-";
const static string unknownkey = "-UNKNOWN-";
const static string seperateKey = "#";
const static string path_separator =
#ifdef _WIN32
    "\\";
#else
    "/";
#endif

typedef std::vector<std::string> CStringVector;

typedef std::vector<std::pair<std::string, std::string> > CTwoStringVector;

class string_less {
 public:
  bool operator()(const string &str1, const string &str2) const {
    int ret = strcmp(str1.c_str(), str2.c_str());
    if (ret < 0)
      return true;
    else
      return false;
  }
};

class LabelScore {
 public:
  int labelId;
  dtype score;

 public:
  LabelScore() {
    labelId = -1;
    score = 0.0;
  }
  LabelScore(int id, dtype value) {
    labelId = id;
    score = value;
  }
};

class LabelScore_Compare {
 public:
  bool operator()(const LabelScore &o1, const LabelScore &o2) const {

    if (o1.score < o2.score)
      return -1;
    else if (o1.score > o2.score)
      return 1;
    else
      return 0;
  }
};

/*==============================================================
 *
 * CSentenceTemplate
 *
 *==============================================================*/

template<typename CSentenceNode>
class CSentenceTemplate : public std::vector<CSentenceNode> {

 public:
  CSentenceTemplate() {
  }
  virtual ~CSentenceTemplate() {
  }
};


//==============================================================

template<typename CSentenceNode>
std::istream &operator>>(std::istream &is, CSentenceTemplate<CSentenceNode> &sent) {
  sent.clear();
  std::string line;
  while (is && line.empty())
    getline(is, line);

  //getline(is, line);

  while (is && !line.empty()) {
    CSentenceNode node;
    std::istringstream iss(line);
    iss >> node;
    sent.push_back(node);
    getline(is, line);
  }
  return is;
}

template<typename CSentenceNode>
std::ostream &operator<<(std::ostream &os, const CSentenceTemplate<CSentenceNode> &sent) {
  for (unsigned i = 0; i < sent.size(); ++i)
    os << sent.at(i) << std::endl;
  os << std::endl;
  return os;
}

void print_time();

char *mystrcat(char *dst, const char *src);

char *mystrdup(const char *src);

int message_callback(void *instance, const char *format, va_list args);

void Free(dtype **p);

int mod(int v1, int v2);

void ones(dtype *p, int length);

void zeros(dtype *p, int length);

dtype logsumexp(dtype a[], int length);

dtype logsumexp(const vector<dtype> &a);

bool isPunc(std::string thePostag);

// start some assumptions, "-*-" is a invalid label.
bool validlabels(const string &curLabel);

string cleanLabel(const string &curLabel);

bool is_start_label(const string &label);

bool is_continue_label(const string &label, const string &startlabel, int distance);

// end some assumptions

int cmpIntIntPairByValue(const pair<int, int> &x, const pair<int, int> &y);

void sortMapbyValue(const unordered_map<int, int> &t_map, vector<pair<int, int> > &t_vec);

void replace_char_by_char(string &str, char c1, char c2);

void split_bychars(const string &str, vector<string> &vec, const char *sep = " ");

// remove the blanks at the begin and end of string
void clean_str(string &str);

bool my_getline(ifstream &inf, string &line);

void str2uint_vec(const vector<string> &vecStr, vector<unsigned int> &vecInt);

void str2int_vec(const vector<string> &vecStr, vector<int> &vecInt);

template<typename A>
string obj2string(const A &a) {
  ostringstream out;
  out << a;
  return out.str();
}

void int2str_vec(const vector<int> &vecInt, vector<string> &vecStr);

void join_bystr(const vector<string> &vec, string &str, const string &sep);

void split_bystr(const string &str, vector<string> &vec, const string &sep);

void split_pair_vector(const vector<pair<int, string> > &vecPair, vector<int> &vecInt, vector<string> &vecStr);

void split_bychar(const string &str, vector<string> &vec, const char separator = ' ');

void string2pair(const string &str, pair<string, string> &pairStr, const char separator = '/');

void convert_to_pair(vector<string> &vecString, vector<pair<string, string> > &vecPair);

void split_to_pair(const string &str, vector<pair<string, string> > &vecPair);

void chomp(string &str);

int common_substr_len(string str1, string str2);

int get_char_index(string &str);

bool is_chinese_char(string &str);

int find_GB_char(const string &str, string wideChar, int begPos);

void split_by_separator(const string &str, vector<string> &vec, const string separator);

//void compute_time()
//{
//  clock_t tick = clock();
//  dtype t = (dtype)tick / CLK_TCK;
//  cout << endl << "The time used: " << t << " seconds." << endl;
//}

string word(string &word_pos);

bool is_ascii_string(string &word);

bool is_startwith(const string &word, const string &prefix);

void remove_beg_end_spaces(string &str);

void split_bystr(const string &str, vector<string> &vec, const char *sep);

string tolowcase(const string &word);

//segmentation index
struct segIndex {
  int start;
  int end;
  string label;
};

void getSegs(const vector<string> &labels, vector<segIndex> &segs);

// vector operations
template<typename A>
void clearVec(vector<vector<A> > &bivec) {
  int count = bivec.size();
  for (int idx = 0; idx < count; idx++) {
    bivec[idx].clear();
  }
  bivec.clear();
}

template<typename A>
void clearVec(vector<vector<vector<A> > > &trivec) {
  int count1, count2;
  count1 = trivec.size();
  for (int idx = 0; idx < count1; idx++) {
    count2 = trivec[idx].size();
    for (int idy = 0; idy < count2; idy++) {
      trivec[idx][idy].clear();
    }
    trivec[idx].clear();
  }
  trivec.clear();
}

template<typename A>
void resizeVec(vector<vector<A> > &bivec, const int &size1, const int &size2) {
  bivec.resize(size1);
  for (int idx = 0; idx < size1; idx++) {
    bivec[idx].resize(size2);
  }
}

template<typename A>
void resizeVec(vector<vector<vector<A> > > &trivec, const int &size1, const int &size2, const int &size3) {
  trivec.resize(size1);
  for (int idx = 0; idx < size1; idx++) {
    trivec[idx].resize(size2);
    for (int idy = 0; idy < size2; idy++) {
      trivec[idx][idy].resize(size3);
    }
  }
}

template<typename A>
void assignVec(vector<A> &univec, const A &a) {
  int count = univec.size();
  for (int idx = 0; idx < count; idx++) {
    univec[idx] = a;
  }
}

template<typename A>
void assignVec(vector<vector<A> > &bivec, const A &a) {
  int count1, count2;
  count1 = bivec.size();
  for (int idx = 0; idx < bivec.size(); idx++) {
    count2 = bivec[idx].size();
    for (int idy = 0; idy < count2; idy++) {
      bivec[idx][idy] = a;
    }
  }
}

template<typename A>
void assignVec(vector<vector<vector<A> > > &trivec, const A &a) {
  int count1, count2, count3;
  count1 = trivec.size();
  for (int idx = 0; idx < count1; idx++) {
    count2 = trivec[idx].size();
    for (int idy = 0; idy < count2; idy++) {
      count3 = trivec[idx][idy].size();
      for (int idz = 0; idz < count3; idz++) {
        trivec[idx][idy][idz] = a;
      }
    }
  }
}

template<typename A>
void addAllItems(vector<A> &target, const vector<A> &sources) {
  int count = sources.size();
  for (int idx = 0; idx < count; idx++) {
    target.push_back(sources[idx]);
  }
}

int cmpStringIntPairByValue(const pair<string, int> &x, const pair<string, int> &y);

template<typename T, typename S>
std::vector<S *> toPointers(std::vector<T> &v, int size) {
  std::vector<S *> pointers;
  for (int i = 0; i < size; ++i) {
    pointers.push_back(&v.at(i));
  }
  return pointers;
}

template<typename T, typename S>
std::vector<S *> toPointers(std::vector<T> &v) {
  return toPointers<T, S>(v, v.size());
}

// for lowercase English only
bool isPunctuation(const std::string &text);

bool isEqual(dtype a, dtype b);

size_t typeSignature(void *p);

#define n3ldg_assert(assertion, message) \
  if (!(assertion)) {\
    std::cerr << message << endl;\
    abort(); \
  }

template<typename T, typename S>
std::vector<T> transferVector(const std::vector<S> &src_vector,
                              const std::function<T(const S &)> &transfer) {
  std::vector<T> result;
  for (const S &src : src_vector) {
    result.push_back(transfer(src));
  }
  return result;
}

template<typename K, typename V>
Json::Value toJson(const unordered_map<K, V> &map) {
  Json::Value result;
  for (auto it = map.begin(); it != map.end(); ++it) {
    result[it->first] = it->second;
  }
  return result;
}

unordered_map<string, int> intMapFromJson(const Json::Value &json);

template<typename T>
Json::Value toJson(const vector<T> &v) {
  Json::Value result;
  for (const T &e : v) {
    result.append(e);
  }
  return result;
}

vector<string> stringVectorFromJson(const Json::Value &json);

#endif

