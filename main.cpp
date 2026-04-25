#include <opencv2/opencv.hpp>

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// === TAKIP SINIFLARI ===
vector<int> IZLENECEK_SINIFLAR = {0, 32, 67};  // 0=Insan, 32=Sports ball, 67=Cell phone
vector<int> REFERANS_SINIFLAR = {39};          // 39=Bottle

// === YAKLASIK Z EKSENI MESAFE AYARLARI ===
const float HEDEF_GERCEK_GENISLIK_CM = 25.0f;
const float KAMERA_ODAK_UZAKLIGI = 600.0f;
const double CONFIDENCE_THRESHOLD = 0.45;
const double NMS_THRESHOLD = 0.45;
const float TRACK_MATCH_DISTANCE_PX = 140.0f;
const int TRACK_MAX_MISSED_FRAMES = 12;
const int PRIMARY_LOCK_MAX_LOST_FRAMES = 8;
const float PRIMARY_LOCK_TRACK_BONUS = 0.35f;
const float PRIMARY_LOCK_CENTER_WEIGHT = 0.25f;
const float PRIMARY_LOCK_AREA_WEIGHT = 0.15f;

struct PrimaryLockState {
    int active_track_id = -1;
    int lost_frames = 0;
};

enum class TakipDurumu {
    Bekleniyor,
    Izleniyor,
    Belirsiz
};

struct Detection {
    int class_id = -1;
    float confidence = 0.0f;
    Rect box;
    int track_id = -1;
    bool kalman_kullanildi = false;
};

struct TimingMetrics {
    double decode_ms = 0.0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double worker_total_ms = 0.0;
};

struct InferenceResult {
    vector<Detection> detections;
    TimingMetrics timings;
};

struct PackageInfo {
    fs::path package_dir;
    fs::path manifest_path;
    fs::path model_path;
    fs::path labels_path;
    int class_count = 0;
};

struct Track {
    int id = -1;
    int class_id = -1;
    int missed_frames = 0;
    float last_confidence = 0.0f;
    Rect estimated_box;
    KalmanFilter filter;
};

bool listedeVarMi(const vector<int>& liste, int id) {
    return find(liste.begin(), liste.end(), id) != liste.end();
}

int hedefOncelikPuani(int class_id) {
    for (size_t i = 0; i < IZLENECEK_SINIFLAR.size(); ++i) {
        if (IZLENECEK_SINIFLAR[i] == class_id) {
            return static_cast<int>(IZLENECEK_SINIFLAR.size() - i);
        }
    }
    return 0;
}

string trim(const string& value) {
    const string whitespace = " \t\r\n";
    const size_t first = value.find_first_not_of(whitespace);
    if (first == string::npos) {
        return "";
    }

    const size_t last = value.find_last_not_of(whitespace);
    return value.substr(first, last - first + 1);
}

string satirOku(HANDLE handle) {
    string line;
    char ch = 0;
    DWORD bytes_read = 0;

    while (true) {
        if (!ReadFile(handle, &ch, 1, &bytes_read, nullptr) || bytes_read == 0) {
            if (line.empty()) {
                throw runtime_error("Worker hattindan veri okunamadi.");
            }
            break;
        }

        if (ch == '\n') {
            break;
        }

        if (ch != '\r') {
            line.push_back(ch);
        }
    }

    return line;
}

void tumunuYaz(HANDLE handle, const void* data, size_t size) {
    const char* cursor = static_cast<const char*>(data);
    size_t remaining = size;

    while (remaining > 0) {
        DWORD written = 0;
        const DWORD chunk = static_cast<DWORD>(std::min<size_t>(remaining, 1 << 20));
        if (!WriteFile(handle, cursor, chunk, &written, nullptr) || written == 0) {
            throw runtime_error("Worker hattina veri yazilamadi.");
        }

        cursor += written;
        remaining -= written;
    }
}

string zamanDamgasiOlustur() {
    auto now = chrono::system_clock::now();
    const auto millis = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;
    time_t now_time = chrono::system_clock::to_time_t(now);
    tm local_tm{};
    localtime_s(&local_tm, &now_time);

    ostringstream output;
    output << put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << '.' << setw(3) << setfill('0') << millis.count();
    return output.str();
}

Point2f kutuMerkezi(const Rect& box) {
    return Point2f(box.x + box.width * 0.5f, box.y + box.height * 0.5f);
}

Rect kutuyuKirp(const Rect& box, const Size& frame_size) {
    Rect frame_rect(0, 0, frame_size.width, frame_size.height);
    return box & frame_rect;
}

Rect stateTenKutuOlustur(const Mat& state, const Size& frame_size) {
    const float cx = state.at<float>(0);
    const float cy = state.at<float>(1);
    const float width = std::max(state.at<float>(2), 1.0f);
    const float height = std::max(state.at<float>(3), 1.0f);

    const int left = cvRound(cx - width * 0.5f);
    const int top = cvRound(cy - height * 0.5f);
    const int right = cvRound(cx + width * 0.5f);
    const int bottom = cvRound(cy + height * 0.5f);

    return kutuyuKirp(Rect(Point(left, top), Point(right, bottom)), frame_size);
}

float hedefSkoru(const Detection& detection, const Size& frame_size, int active_track_id) {
    const Point2f center = kutuMerkezi(detection.box);
    const Point2f frame_center(frame_size.width * 0.5f, frame_size.height * 0.5f);
    const float max_distance = std::sqrt(frame_center.x * frame_center.x + frame_center.y * frame_center.y);
    const float center_distance = static_cast<float>(norm(center - frame_center));
    const float center_score = 1.0f - std::min(center_distance / std::max(max_distance, 1.0f), 1.0f);
    const float area_score = std::min(
        static_cast<float>(detection.box.area()) / std::max(frame_size.area() * 0.2f, 1.0f),
        1.0f);
    const float continuity_bonus = (detection.track_id == active_track_id) ? PRIMARY_LOCK_TRACK_BONUS : 0.0f;
    const float class_bonus = 0.05f * hedefOncelikPuani(detection.class_id);

    return detection.confidence + center_score * PRIMARY_LOCK_CENTER_WEIGHT + area_score * PRIMARY_LOCK_AREA_WEIGHT +
           continuity_bonus + class_bonus;
}

string takipDurumuMetni(TakipDurumu durum) {
    switch (durum) {
        case TakipDurumu::Izleniyor:
            return "IZLENIYOR";
        case TakipDurumu::Belirsiz:
            return "BELIRSIZ";
        default:
            return "BEKLENIYOR";
    }
}

int birincilHedefiSec(const vector<Detection>& detections, const Size& frame_size, const PrimaryLockState& lock_state) {
    float best_score = -1.0f;
    int best_index = -1;

    for (size_t i = 0; i < detections.size(); ++i) {
        if (!listedeVarMi(IZLENECEK_SINIFLAR, detections[i].class_id)) {
            continue;
        }

        const float score = hedefSkoru(detections[i], frame_size, lock_state.active_track_id);
        if (score > best_score) {
            best_score = score;
            best_index = static_cast<int>(i);
        }
    }

    return best_index;
}

KalmanFilter kalmanFiltresiOlustur(const Rect& initial_box) {
    KalmanFilter filter(8, 4, 0, CV_32F);

    setIdentity(filter.transitionMatrix);
    filter.transitionMatrix.at<float>(0, 4) = 1.0f;
    filter.transitionMatrix.at<float>(1, 5) = 1.0f;
    filter.transitionMatrix.at<float>(2, 6) = 1.0f;
    filter.transitionMatrix.at<float>(3, 7) = 1.0f;

    filter.measurementMatrix = Mat::zeros(4, 8, CV_32F);
    filter.measurementMatrix.at<float>(0, 0) = 1.0f;
    filter.measurementMatrix.at<float>(1, 1) = 1.0f;
    filter.measurementMatrix.at<float>(2, 2) = 1.0f;
    filter.measurementMatrix.at<float>(3, 3) = 1.0f;

    setIdentity(filter.processNoiseCov, Scalar::all(1e-2));
    setIdentity(filter.measurementNoiseCov, Scalar::all(7e-1));
    setIdentity(filter.errorCovPost, Scalar::all(1.0f));
    setIdentity(filter.errorCovPre, Scalar::all(1.0f));

    filter.statePost = Mat::zeros(8, 1, CV_32F);
    filter.statePost.at<float>(0) = initial_box.x + initial_box.width * 0.5f;
    filter.statePost.at<float>(1) = initial_box.y + initial_box.height * 0.5f;
    filter.statePost.at<float>(2) = static_cast<float>(std::max(initial_box.width, 1));
    filter.statePost.at<float>(3) = static_cast<float>(std::max(initial_box.height, 1));

    filter.statePre = filter.statePost.clone();
    return filter;
}

class DetectionTracker {
public:
    vector<Detection> guncelle(const vector<Detection>& detections, const Size& frame_size) {
        vector<Detection> filtered = detections;
        vector<Rect> predicted_boxes(tracks_.size());

        for (size_t i = 0; i < tracks_.size(); ++i) {
            Mat prediction = tracks_[i].filter.predict();
            tracks_[i].estimated_box = stateTenKutuOlustur(prediction, frame_size);
            predicted_boxes[i] = tracks_[i].estimated_box;
            tracks_[i].missed_frames++;
        }

        vector<bool> track_used(tracks_.size(), false);

        for (Detection& detection : filtered) {
            const Point2f detection_center = kutuMerkezi(detection.box);
            float best_distance = std::numeric_limits<float>::max();
            int best_track_index = -1;

            for (size_t i = 0; i < tracks_.size(); ++i) {
                if (track_used[i] || tracks_[i].class_id != detection.class_id) {
                    continue;
                }

                const Point2f predicted_center = kutuMerkezi(predicted_boxes[i]);
                const float distance = static_cast<float>(norm(detection_center - predicted_center));
                if (distance < best_distance && distance <= TRACK_MATCH_DISTANCE_PX) {
                    best_distance = distance;
                    best_track_index = static_cast<int>(i);
                }
            }

            if (best_track_index >= 0) {
                Track& track = tracks_[best_track_index];
                Mat measurement(4, 1, CV_32F);
                measurement.at<float>(0) = detection_center.x;
                measurement.at<float>(1) = detection_center.y;
                measurement.at<float>(2) = static_cast<float>(std::max(detection.box.width, 1));
                measurement.at<float>(3) = static_cast<float>(std::max(detection.box.height, 1));

                Mat estimated = track.filter.correct(measurement);
                track.estimated_box = stateTenKutuOlustur(estimated, frame_size);
                track.last_confidence = detection.confidence;
                track.missed_frames = 0;

                detection.box = track.estimated_box;
                detection.track_id = track.id;
                detection.kalman_kullanildi = true;
                track_used[best_track_index] = true;
            } else {
                Track new_track;
                new_track.id = next_track_id_++;
                new_track.class_id = detection.class_id;
                new_track.last_confidence = detection.confidence;
                new_track.filter = kalmanFiltresiOlustur(detection.box);
                new_track.estimated_box = kutuyuKirp(detection.box, frame_size);
                tracks_.push_back(new_track);
                track_used.push_back(true);
                predicted_boxes.push_back(new_track.estimated_box);

                detection.box = new_track.estimated_box;
                detection.track_id = new_track.id;
                detection.kalman_kullanildi = true;
            }
        }

        tracks_.erase(
            remove_if(tracks_.begin(),
                      tracks_.end(),
                      [](const Track& track) { return track.missed_frames > TRACK_MAX_MISSED_FRAMES; }),
            tracks_.end());

        return filtered;
    }

private:
    int next_track_id_ = 1;
    vector<Track> tracks_;
};

vector<string> etiketleriYukle(const fs::path& labels_path) {
    ifstream input(labels_path);
    if (!input.is_open()) {
        throw runtime_error("labels.txt acilamadi: " + labels_path.string());
    }

    vector<string> labels;
    string line;
    while (getline(input, line)) {
        const string cleaned = trim(line);
        if (!cleaned.empty()) {
            labels.push_back(cleaned);
        }
    }

    return labels;
}

PackageInfo paketiYukle(const fs::path& package_dir) {
    PackageInfo info;
    info.package_dir = fs::absolute(package_dir);
    info.manifest_path = info.package_dir / "model-manifest.json";

    FileStorage manifest(info.manifest_path.string(), FileStorage::READ | FileStorage::FORMAT_JSON);
    if (!manifest.isOpened()) {
        throw runtime_error("model-manifest.json acilamadi: " + info.manifest_path.string());
    }

    string model_name;
    string labels_name;
    FileNode files_node = manifest["files"];
    files_node["model"] >> model_name;
    files_node["labels"] >> labels_name;

    FileNode model_node = manifest["model"];
    model_node["class_count"] >> info.class_count;

    if (model_name.empty() || labels_name.empty()) {
        throw runtime_error("Manifest icindeki files alani eksik.");
    }

    info.model_path = info.package_dir / model_name;
    info.labels_path = info.package_dir / labels_name;

    if (!fs::exists(info.model_path)) {
        throw runtime_error("Model dosyasi bulunamadi: " + info.model_path.string());
    }

    if (!fs::exists(info.labels_path)) {
        throw runtime_error("Etiket dosyasi bulunamadi: " + info.labels_path.string());
    }

    return info;
}

wstring pythonYoluBul() {
    char env_buffer[1024] = {};
    const DWORD env_length = GetEnvironmentVariableA("KORGAN_PYTHON", env_buffer, static_cast<DWORD>(sizeof(env_buffer)));
    if (env_length > 0 && env_length < sizeof(env_buffer)) {
        fs::path env_python(env_buffer);
        if (fs::exists(env_python)) {
            return env_python.wstring();
        }
    }

    const vector<fs::path> candidates = {
        fs::path("C:/Users/askan/AppData/Local/Programs/Python/Python312/python.exe"),
        fs::path("python.exe")
    };

    for (const auto& candidate : candidates) {
        if (candidate.filename() == "python.exe" || fs::exists(candidate)) {
            return candidate.wstring();
        }
    }

    throw runtime_error("Python bulunamadi. KORGAN_PYTHON ortam degiskenini ayarlayin.");
}

class OrtWorkerClient {
public:
    OrtWorkerClient(const fs::path& package_dir) {
        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;

        HANDLE child_stdout_read = nullptr;
        HANDLE child_stdout_write = nullptr;
        HANDLE child_stdin_read = nullptr;
        HANDLE child_stdin_write = nullptr;

        if (!CreatePipe(&child_stdout_read, &child_stdout_write, &sa, 0)) {
            throw runtime_error("STDOUT pipe olusturulamadi.");
        }
        if (!SetHandleInformation(child_stdout_read, HANDLE_FLAG_INHERIT, 0)) {
            CloseHandle(child_stdout_read);
            CloseHandle(child_stdout_write);
            throw runtime_error("STDOUT pipe handle ayarlanamadi.");
        }

        if (!CreatePipe(&child_stdin_read, &child_stdin_write, &sa, 0)) {
            CloseHandle(child_stdout_read);
            CloseHandle(child_stdout_write);
            throw runtime_error("STDIN pipe olusturulamadi.");
        }
        if (!SetHandleInformation(child_stdin_write, HANDLE_FLAG_INHERIT, 0)) {
            CloseHandle(child_stdout_read);
            CloseHandle(child_stdout_write);
            CloseHandle(child_stdin_read);
            CloseHandle(child_stdin_write);
            throw runtime_error("STDIN pipe handle ayarlanamadi.");
        }

        const fs::path worker_path = fs::absolute("scripts/ort_worker.py");
        if (!fs::exists(worker_path)) {
            CloseHandle(child_stdout_read);
            CloseHandle(child_stdout_write);
            CloseHandle(child_stdin_read);
            CloseHandle(child_stdin_write);
            throw runtime_error("Worker script bulunamadi: " + worker_path.string());
        }

        const wstring python_path = pythonYoluBul();
        const wstring command = L"\"" + python_path + L"\" \"" + worker_path.wstring() +
                                L"\" --package \"" + fs::absolute(package_dir).wstring() + L"\"";

        STARTUPINFOW startup_info{};
        startup_info.cb = sizeof(startup_info);
        startup_info.dwFlags = STARTF_USESTDHANDLES;
        startup_info.hStdInput = child_stdin_read;
        startup_info.hStdOutput = child_stdout_write;
        startup_info.hStdError = child_stdout_write;

        vector<wchar_t> mutable_command(command.begin(), command.end());
        mutable_command.push_back(L'\0');

        ZeroMemory(&process_info_, sizeof(process_info_));
        if (!CreateProcessW(nullptr,
                            mutable_command.data(),
                            nullptr,
                            nullptr,
                            TRUE,
                            CREATE_NO_WINDOW,
                            nullptr,
                            nullptr,
                            &startup_info,
                            &process_info_)) {
            CloseHandle(child_stdout_read);
            CloseHandle(child_stdout_write);
            CloseHandle(child_stdin_read);
            CloseHandle(child_stdin_write);
            throw runtime_error("Python worker baslatilamadi.");
        }

        CloseHandle(child_stdout_write);
        CloseHandle(child_stdin_read);
        stdout_read_ = child_stdout_read;
        stdin_write_ = child_stdin_write;

        const string ready_line = satirOku(stdout_read_);
        if (ready_line.rfind("READY", 0) != 0) {
            kapat();
            throw runtime_error("Worker hazirlik cevabi alinamadi: " + ready_line);
        }
    }

    OrtWorkerClient(const OrtWorkerClient&) = delete;
    OrtWorkerClient& operator=(const OrtWorkerClient&) = delete;

    ~OrtWorkerClient() {
        try {
            if (stdin_write_ != nullptr) {
                const string quit = "QUIT\n";
                tumunuYaz(stdin_write_, quit.data(), quit.size());
            }
        } catch (...) {
        }
        kapat();
    }

    InferenceResult tahminEt(const Mat& frame) {
        vector<uchar> jpeg_buffer;
        vector<int> jpeg_params = {IMWRITE_JPEG_QUALITY, 90};
        if (!imencode(".jpg", frame, jpeg_buffer, jpeg_params)) {
            throw runtime_error("Frame JPEG formatina donusturulemedi.");
        }

        ostringstream header;
        header << "FRAME " << jpeg_buffer.size() << "\n";
        const string header_str = header.str();
        tumunuYaz(stdin_write_, header_str.data(), header_str.size());
        tumunuYaz(stdin_write_, jpeg_buffer.data(), jpeg_buffer.size());

        const string response = satirOku(stdout_read_);
        return cevapAyikla(response);
    }

private:
    HANDLE stdout_read_ = nullptr;
    HANDLE stdin_write_ = nullptr;
    PROCESS_INFORMATION process_info_{};

    void kapat() {
        if (stdin_write_ != nullptr) {
            CloseHandle(stdin_write_);
            stdin_write_ = nullptr;
        }
        if (stdout_read_ != nullptr) {
            CloseHandle(stdout_read_);
            stdout_read_ = nullptr;
        }
        if (process_info_.hThread != nullptr) {
            CloseHandle(process_info_.hThread);
            process_info_.hThread = nullptr;
        }
        if (process_info_.hProcess != nullptr) {
            WaitForSingleObject(process_info_.hProcess, 1500);
            CloseHandle(process_info_.hProcess);
            process_info_.hProcess = nullptr;
        }
    }

    InferenceResult cevapAyikla(const string& response) {
        FileStorage fs(response, FileStorage::READ | FileStorage::MEMORY | FileStorage::FORMAT_JSON);
        if (!fs.isOpened()) {
            throw runtime_error("Worker cevabi JSON olarak okunamadi: " + response);
        }

        string error_message;
        fs["error"] >> error_message;
        if (!error_message.empty()) {
            throw runtime_error("Worker hata verdi: " + error_message);
        }

        InferenceResult result;
        FileNode timings_node = fs["timings_ms"];
        if (!timings_node.empty()) {
            timings_node["decode"] >> result.timings.decode_ms;
            timings_node["preprocess"] >> result.timings.preprocess_ms;
            timings_node["inference"] >> result.timings.inference_ms;
            timings_node["postprocess"] >> result.timings.postprocess_ms;
            timings_node["worker_total"] >> result.timings.worker_total_ms;
        }

        FileNode detections_node = fs["detections"];
        if (detections_node.type() != FileNode::SEQ) {
            return result;
        }

        for (auto it = detections_node.begin(); it != detections_node.end(); ++it) {
            Detection det;
            (*it)["class_id"] >> det.class_id;
            (*it)["confidence"] >> det.confidence;

            vector<int> bbox;
            (*it)["bbox_xyxy"] >> bbox;
            if (bbox.size() != 4) {
                continue;
            }

            const int left = bbox[0];
            const int top = bbox[1];
            const int right = bbox[2];
            const int bottom = bbox[3];
            det.box = Rect(Point(left, top), Point(right, bottom));
            result.detections.push_back(det);
        }

        return result;
    }
};

int main(int argc, char** argv) {
    try {
        const fs::path package_dir = (argc > 1) ? fs::path(argv[1]) : fs::path("deliverables/onnxruntime_cpu_package");
        const PackageInfo package_info = paketiYukle(package_dir);
        const vector<string> labels = etiketleriYukle(package_info.labels_path);

        if (package_info.class_count > 0 && static_cast<int>(labels.size()) != package_info.class_count) {
            throw runtime_error("labels.txt sinif sayisi manifest ile uyusmuyor.");
        }

        cout << "==========================================" << endl;
        cout << "[SISTEM] ORT CPU destekli takip paketi yukleniyor..." << endl;
        cout << "[PAKET] " << package_info.package_dir.string() << endl;
        cout << "[MODEL] " << package_info.model_path.filename().string() << endl;
        cout << "[ETIKET] " << labels.size() << " sinif" << endl;
        cout << "==========================================" << endl;

        ofstream log_dosyasi("telemetri_log.txt", ios::app);
        if (!log_dosyasi.is_open()) {
            throw runtime_error("telemetri_log.txt acilamadi.");
        }
        log_dosyasi << "--- YENI ORT CPU OPERASYONU BASLADI ---" << endl;

        OrtWorkerClient worker(package_info.package_dir);
        DetectionTracker tracker;
        PrimaryLockState primary_lock;

        VideoCapture cap(0);
        if (!cap.isOpened()) {
            throw runtime_error("Kamera acilamadi.");
        }

        Mat frame;
        double smoothed_fps = 0.0;
        size_t previous_terminal_status_length = 0;
        while (true) {
            const auto frame_start = chrono::steady_clock::now();
            cap.read(frame);
            if (frame.empty()) {
                break;
            }

            const auto worker_start = chrono::steady_clock::now();
            const InferenceResult inference_result = worker.tahminEt(frame);
            const auto worker_end = chrono::steady_clock::now();
            const vector<Detection> detections = tracker.guncelle(inference_result.detections, frame.size());
            const auto tracker_end = chrono::steady_clock::now();
            const int primary_index = birincilHedefiSec(detections, frame.size(), primary_lock);

            if (primary_index >= 0) {
                primary_lock.active_track_id = detections[primary_index].track_id;
                primary_lock.lost_frames = 0;
            } else if (primary_lock.active_track_id >= 0) {
                primary_lock.lost_frames++;
                if (primary_lock.lost_frames > PRIMARY_LOCK_MAX_LOST_FRAMES) {
                    primary_lock.active_track_id = -1;
                    primary_lock.lost_frames = 0;
                }
            }

            bool primary_telemetry_visible = false;
            int primary_cx = 0;
            int primary_cy = 0;
            int primary_z_cm = 0;
            int primary_target_id = -1;
            int primary_track_id = -1;
            string primary_label = "-";
            string primary_confidence = "0.00";
            float primary_confidence_value = 0.0f;

            for (size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
                const auto& detection = detections[detection_index];
                Rect box = detection.box & Rect(0, 0, frame.cols, frame.rows);
                if (box.width <= 0 || box.height <= 0) {
                    continue;
                }

                const int nesne_id = detection.class_id;
                const string label = (nesne_id >= 0 && nesne_id < static_cast<int>(labels.size()))
                                         ? labels[nesne_id]
                                         : ("class_" + to_string(nesne_id));

                const int cx = box.x + box.width / 2;
                const int cy = box.y + box.height / 2;
                const int mesafe_z_cm = static_cast<int>((HEDEF_GERCEK_GENISLIK_CM * KAMERA_ODAK_UZAKLIGI) /
                                                         std::max(box.width, 1));

                const string confidence_text = format("%.2f", detection.confidence);
                const string takip_text = "T" + to_string(detection.track_id) + (detection.kalman_kullanildi ? " KF" : "");
                const bool primary_lock_active =
                    primary_index >= 0 && static_cast<int>(detection_index) == primary_index &&
                    detection.track_id == primary_lock.active_track_id;

                if (listedeVarMi(REFERANS_SINIFLAR, nesne_id)) {
                    rectangle(frame, box, Scalar(255, 0, 0), 2);
                    putText(frame,
                            "REFERANS " + label + " " + to_string(mesafe_z_cm) + " cm",
                            Point(box.x, std::max(box.y - 10, 20)),
                            FONT_HERSHEY_SIMPLEX,
                            0.55,
                            Scalar(255, 0, 0),
                            2);
                    putText(frame,
                            takip_text,
                            Point(box.x, std::min(box.y + box.height + 18, frame.rows - 10)),
                            FONT_HERSHEY_SIMPLEX,
                            0.45,
                            Scalar(255, 180, 0),
                            1);
                } else if (listedeVarMi(IZLENECEK_SINIFLAR, nesne_id)) {
                    const Scalar tracked_color = primary_lock_active ? Scalar(0, 170, 255) : Scalar(0, 220, 255);
                    const Scalar center_color = primary_lock_active ? Scalar(0, 255, 0) : Scalar(0, 220, 255);
                    rectangle(frame, box, tracked_color, primary_lock_active ? 2 : 1);
                    circle(frame, Point(cx, cy), primary_lock_active ? 5 : 3, center_color, -1);

                    if (primary_lock_active) {
                        primary_telemetry_visible = true;
                        primary_cx = cx;
                        primary_cy = cy;
                        primary_z_cm = mesafe_z_cm;
                        primary_target_id = nesne_id;
                        primary_track_id = detection.track_id;
                        primary_label = label;
                        primary_confidence = confidence_text;
                        primary_confidence_value = detection.confidence;

                        putText(frame,
                                "ODAK " + label + " @" + confidence_text + " Z:" + to_string(mesafe_z_cm) + "cm",
                                Point(box.x, std::max(box.y - 10, 20)),
                                FONT_HERSHEY_SIMPLEX,
                                0.55,
                                Scalar(0, 170, 255),
                                2);
                        putText(frame,
                                "X:" + to_string(cx) + " Y:" + to_string(cy) + " Z:" + to_string(mesafe_z_cm),
                                Point(box.x, std::min(box.y + box.height + 20, frame.rows - 10)),
                                FONT_HERSHEY_SIMPLEX,
                                0.5,
                                Scalar(0, 255, 0),
                                2);
                        putText(frame,
                                takip_text + " ODAK",
                                Point(std::max(box.x - 2, 0), std::min(box.y + box.height + 40, frame.rows - 10)),
                                FONT_HERSHEY_SIMPLEX,
                                0.45,
                                Scalar(255, 220, 0),
                                1);

                        log_dosyasi << "[" << zamanDamgasiOlustur() << "] GOZLEM -> X:" << cx
                                    << " | Y:" << cy << " | Z:" << mesafe_z_cm << "cm | ID:" << nesne_id
                                    << " | LABEL:" << label << " | CONF:" << confidence_text
                                    << " | TRACK:" << detection.track_id << " | MODE:ODAK"
                                    << " | LATENCY_WORKER:" << format("%.2f", inference_result.timings.worker_total_ms)
                                    << "ms | LATENCY_INFER:" << format("%.2f", inference_result.timings.inference_ms)
                                    << "ms" << endl;
                    } else {
                        putText(frame,
                                "IZLEME " + label + " @" + confidence_text + " " + takip_text,
                                Point(box.x, std::max(box.y - 8, 20)),
                                FONT_HERSHEY_SIMPLEX,
                                0.45,
                                Scalar(0, 180, 255),
                                1);
                    }
                } else {
                    rectangle(frame, box, Scalar(0, 255, 255), 1);
                    putText(frame,
                            label + " @" + confidence_text + " " + takip_text,
                            Point(box.x, std::max(box.y - 8, 20)),
                            FONT_HERSHEY_SIMPLEX,
                            0.45,
                            Scalar(0, 255, 255),
                            1);
                }
            }

            putText(frame,
                    "ORT CPU  conf>" + format("%.2f", CONFIDENCE_THRESHOLD) + "  nms>" + format("%.2f", NMS_THRESHOLD) +
                        "  kalman:on",
                    Point(20, 30),
                    FONT_HERSHEY_SIMPLEX,
                    0.6,
                    Scalar(255, 255, 255),
                    2);
            putText(frame,
                    primary_lock.active_track_id >= 0
                        ? "AKTIF TAKIP: T" + to_string(primary_lock.active_track_id)
                        : "AKTIF TAKIP: BEKLENIYOR",
                    Point(20, 58),
                    FONT_HERSHEY_SIMPLEX,
                    0.55,
                    primary_lock.active_track_id >= 0 ? Scalar(0, 255, 0) : Scalar(0, 220, 255),
                    2);

            const auto overlay_end = chrono::steady_clock::now();
            const double worker_round_trip_ms =
                chrono::duration<double, milli>(worker_end - worker_start).count();
            const double tracker_ms = chrono::duration<double, milli>(tracker_end - worker_end).count();
            const double render_ms = chrono::duration<double, milli>(overlay_end - tracker_end).count();
            const double frame_total_ms = chrono::duration<double, milli>(overlay_end - frame_start).count();
            const double instantaneous_fps = frame_total_ms > 0.0 ? 1000.0 / frame_total_ms : 0.0;
            smoothed_fps = (smoothed_fps <= 0.0) ? instantaneous_fps : (smoothed_fps * 0.9 + instantaneous_fps * 0.1);

            const TakipDurumu takip_durumu = primary_telemetry_visible
                                                 ? TakipDurumu::Izleniyor
                                                 : (primary_lock.active_track_id >= 0 ? TakipDurumu::Belirsiz
                                                                                      : TakipDurumu::Bekleniyor);
            const double track_health =
                primary_telemetry_visible
                    ? std::min(100.0, std::max(0.0, static_cast<double>(primary_confidence_value) * 100.0))
                    : std::max(0.0,
                               100.0 -
                                   (primary_lock.lost_frames * 100.0 / std::max(PRIMARY_LOCK_MAX_LOST_FRAMES, 1)));

            const string telemetry_line =
                primary_telemetry_visible
                    ? "TS:" + zamanDamgasiOlustur() + " DURUM:" + takipDurumuMetni(takip_durumu) +
                          " X:" + to_string(primary_cx) + " Y:" + to_string(primary_cy) +
                          " Z:" + to_string(primary_z_cm) + "cm ID:" + to_string(primary_target_id) +
                          " " + primary_label + " T:" + to_string(primary_track_id) +
                          " CONF:" + primary_confidence + " HEALTH:" + format("%.0f", track_health) + "%"
                    : "TS:" + zamanDamgasiOlustur() + " DURUM:" + takipDurumuMetni(takip_durumu) +
                          " T:" + (primary_lock.active_track_id >= 0 ? to_string(primary_lock.active_track_id) : "-") +
                          " KAYIP:" + to_string(primary_lock.lost_frames) +
                          " HEALTH:" + format("%.0f", track_health) + "%";
            const string latency_line =
                "GECIKME RT:" + format("%.1f", worker_round_trip_ms) + "ms" +
                " DEC:" + format("%.1f", inference_result.timings.decode_ms) + "ms" +
                " PRE:" + format("%.1f", inference_result.timings.preprocess_ms) + "ms" +
                " INF:" + format("%.1f", inference_result.timings.inference_ms) + "ms" +
                " POST:" + format("%.1f", inference_result.timings.postprocess_ms) + "ms" +
                " TRK:" + format("%.1f", tracker_ms) + "ms" +
                " DRW:" + format("%.1f", render_ms) + "ms" +
                " FRM:" + format("%.1f", frame_total_ms) + "ms";
            const string terminal_status = "[TEL] " + telemetry_line + " | " + latency_line;
            const size_t clear_padding =
                (previous_terminal_status_length > terminal_status.size())
                    ? (previous_terminal_status_length - terminal_status.size())
                    : 0;
            cout << '\r' << terminal_status << string(clear_padding, ' ') << flush;
            previous_terminal_status_length = terminal_status.size();

            const string fps_text = "FPS " + format("%.1f", smoothed_fps);
            int baseline = 0;
            const Size fps_size = getTextSize(fps_text, FONT_HERSHEY_SIMPLEX, 0.65, 2, &baseline);
            const Rect fps_box(frame.cols - fps_size.width - 28, 14, fps_size.width + 16, fps_size.height + 14);
            rectangle(frame, fps_box, Scalar(20, 20, 20), FILLED);
            rectangle(frame, fps_box, Scalar(0, 200, 255), 1);
            putText(frame,
                    fps_text,
                    Point(fps_box.x + 8, fps_box.y + fps_size.height + 2),
                    FONT_HERSHEY_SIMPLEX,
                    0.65,
                    Scalar(0, 255, 255),
                    2);

            imshow("Kamera Takip Ekrani - ORT CPU Paket Modu", frame);
            if (waitKey(1) == 27) {
                break;
            }
        }

        if (previous_terminal_status_length > 0) {
            cout << endl;
        }
        cap.release();
        destroyAllWindows();
        return 0;
    } catch (const exception& ex) {
        cerr << "[HATA] " << ex.what() << endl;
        return 1;
    }
}
