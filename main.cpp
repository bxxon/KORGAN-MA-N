#include <opencv2/opencv.hpp>

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

struct PrimaryLockState {
    int active_track_id = -1;
    int lost_frames = 0;
};

enum class TakipDurumu {
    Bekleniyor,
    KilitHazir,
    Izleniyor,
    Belirsiz
};

struct TargetProfile {
    string ad;
    string etiket;
    string gorunen_ad;
    float gercek_genislik_cm = 25.0f;
    float min_kilit_mesafe_m = 0.0f;
    float max_kilit_mesafe_m = 1000.0f;
    int oncelik = 1;
    int resolved_class_id = -1;
};

struct Detection {
    int class_id = -1;
    float confidence = 0.0f;
    Rect box;
    int track_id = -1;
    bool kalman_kullanildi = false;
    bool predicted_only = false;
    bool low_confidence_match = false;
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

struct RuntimeConfig {
    vector<int> izlenecek_siniflar = {0, 32, 67};
    vector<int> referans_siniflar = {39};
    vector<TargetProfile> hedef_profilleri;

    float hedef_gercek_genislik_cm = 25.0f;
    float kamera_odak_uzakligi = 600.0f;

    double detector_conf_threshold = 0.15;
    double tracker_high_conf_threshold = 0.45;
    double tracker_low_conf_threshold = 0.15;
    double nms_threshold = 0.45;

    float track_match_iou_threshold = 0.30f;
    float low_conf_match_iou_threshold = 0.20f;
    int track_max_missed_frames = 12;
    int track_visible_missed_frames = 4;
    int track_confirm_hits = 2;

    int primary_lock_max_lost_frames = 8;
    float primary_lock_track_bonus = 0.35f;
    float primary_lock_center_weight = 0.25f;
    float primary_lock_area_weight = 0.15f;
    bool kilit_icin_mesafe_zorunlu = false;
    float kilit_aralik_bonus = 0.20f;

    int camera_index = 0;
    bool show_window = true;
    bool enable_recording = false;
    fs::path record_dir = "recordings";
    fs::path log_dir = "logs";

    fs::path package_dir = "deliverables/onnxruntime_cpu_package";
    fs::path worker_script = "scripts/ort_worker.py";
    fs::path python_path;
    int worker_jpeg_quality = 90;
    bool worker_auto_restart = true;
    int worker_max_restarts = 3;

    int camera_reopen_attempts = 4;
    int camera_reopen_delay_ms = 750;
    bool log_every_frame = true;
};

struct AppOptions {
    fs::path config_path = "config/runtime_config.json";
    fs::path package_override;
    fs::path python_override;
    fs::path log_dir_override;
    fs::path record_dir_override;
    int camera_override = -1;
    bool headless = false;
    bool force_show = false;
    bool record_override_set = false;
    bool record_override_value = false;
};

struct WorkerLaunchOptions {
    fs::path package_dir;
    fs::path worker_script;
    fs::path python_path;
    double detector_conf_threshold = 0.15;
    double nms_threshold = 0.45;
    int jpeg_quality = 90;
};

struct Track {
    int id = -1;
    int class_id = -1;
    int hits = 0;
    int missed_frames = 0;
    bool confirmed = false;
    float last_confidence = 0.0f;
    Rect estimated_box;
    KalmanFilter filter;
};

struct MatchCandidate {
    size_t track_index = 0;
    size_t detection_index = 0;
    float score = 0.0f;
};

bool listedeVarMi(const vector<int>& liste, int id) {
    return find(liste.begin(), liste.end(), id) != liste.end();
}

string kucukHarf(string value) {
    transform(value.begin(),
              value.end(),
              value.begin(),
              [](unsigned char ch) { return static_cast<char>(tolower(ch)); });
    return value;
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

bool basliyorMu(const string& value, const string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

string jsonKacis(const string& value) {
    ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << hex << setw(4) << setfill('0') << static_cast<int>(ch) << dec;
                } else {
                    out << ch;
                }
                break;
        }
    }
    return out.str();
}

string jsonBool(bool value) {
    return value ? "true" : "false";
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

string dosyaAdiIcinZamanDamgasi() {
    time_t now_time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    tm local_tm{};
    localtime_s(&local_tm, &now_time);

    ostringstream output;
    output << put_time(&local_tm, "%Y%m%d_%H%M%S");
    return output.str();
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

float kutuIou(const Rect& a, const Rect& b) {
    const Rect intersection = a & b;
    if (intersection.width <= 0 || intersection.height <= 0) {
        return 0.0f;
    }

    const float inter_area = static_cast<float>(intersection.area());
    const float union_area = static_cast<float>(a.area() + b.area()) - inter_area;
    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return inter_area / union_area;
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

int hedefOncelikPuani(const RuntimeConfig& config, int class_id) {
    for (const auto& profile : config.hedef_profilleri) {
        if (profile.resolved_class_id == class_id) {
            return max(profile.oncelik, 0);
        }
    }

    for (size_t i = 0; i < config.izlenecek_siniflar.size(); ++i) {
        if (config.izlenecek_siniflar[i] == class_id) {
            return static_cast<int>(config.izlenecek_siniflar.size() - i);
        }
    }
    return 0;
}

const TargetProfile* hedefProfiliBul(const RuntimeConfig& config, int class_id) {
    for (const auto& profile : config.hedef_profilleri) {
        if (profile.resolved_class_id == class_id) {
            return &profile;
        }
    }
    return nullptr;
}

int hedefMesafesiCmHesapla(const RuntimeConfig& config, const Detection& detection, const TargetProfile* profile) {
    const float gercek_genislik_cm =
        (profile != nullptr && profile->gercek_genislik_cm > 0.0f) ? profile->gercek_genislik_cm
                                                                    : config.hedef_gercek_genislik_cm;
    return static_cast<int>((gercek_genislik_cm * config.kamera_odak_uzakligi) / std::max(detection.box.width, 1));
}

bool hedefKilitMesafesindeMi(const TargetProfile* profile, int mesafe_cm) {
    if (profile == nullptr) {
        return true;
    }

    const float mesafe_m = mesafe_cm / 100.0f;
    return mesafe_m >= profile->min_kilit_mesafe_m && mesafe_m <= profile->max_kilit_mesafe_m;
}

float hedefSkoru(const RuntimeConfig& config,
                 const Detection& detection,
                 const Size& frame_size,
                 int active_track_id) {
    const Point2f center = kutuMerkezi(detection.box);
    const Point2f frame_center(frame_size.width * 0.5f, frame_size.height * 0.5f);
    const float max_distance = std::sqrt(frame_center.x * frame_center.x + frame_center.y * frame_center.y);
    const float center_distance = static_cast<float>(norm(center - frame_center));
    const float center_score = 1.0f - std::min(center_distance / std::max(max_distance, 1.0f), 1.0f);
    const float area_score =
        std::min(static_cast<float>(detection.box.area()) /
                     std::max(static_cast<float>(frame_size.area()) * 0.2f, 1.0f),
                 1.0f);
    const TargetProfile* profile = hedefProfiliBul(config, detection.class_id);
    const int mesafe_cm = hedefMesafesiCmHesapla(config, detection, profile);
    const bool lock_in_range = hedefKilitMesafesindeMi(profile, mesafe_cm);
    const float continuity_bonus = (detection.track_id == active_track_id) ? config.primary_lock_track_bonus : 0.0f;
    const float class_bonus = 0.05f * hedefOncelikPuani(config, detection.class_id);
    const float predicted_penalty = detection.predicted_only ? 0.35f : 0.0f;
    const float range_bonus = lock_in_range ? config.kilit_aralik_bonus : (profile != nullptr ? -0.15f : 0.0f);

    return detection.confidence + center_score * config.primary_lock_center_weight +
           area_score * config.primary_lock_area_weight + continuity_bonus + class_bonus + range_bonus -
           predicted_penalty;
}

string takipDurumuMetni(TakipDurumu durum) {
    switch (durum) {
        case TakipDurumu::KilitHazir:
            return "KILIT_HAZIR";
        case TakipDurumu::Izleniyor:
            return "IZLENIYOR";
        case TakipDurumu::Belirsiz:
            return "BELIRSIZ";
        default:
            return "BEKLENIYOR";
    }
}

int birincilHedefiSec(const RuntimeConfig& config,
                      const vector<Detection>& detections,
                      const Size& frame_size,
                      const PrimaryLockState& lock_state) {
    float best_score = -numeric_limits<float>::max();
    int best_index = -1;

    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].predicted_only || !listedeVarMi(config.izlenecek_siniflar, detections[i].class_id)) {
            continue;
        }

        const TargetProfile* profile = hedefProfiliBul(config, detections[i].class_id);
        if (config.kilit_icin_mesafe_zorunlu && profile != nullptr &&
            !hedefKilitMesafesindeMi(profile, hedefMesafesiCmHesapla(config, detections[i], profile))) {
            continue;
        }

        const float score = hedefSkoru(config, detections[i], frame_size, lock_state.active_track_id);
        if (score > best_score) {
            best_score = score;
            best_index = static_cast<int>(i);
        }
    }

    return best_index;
}

template <typename T>
void alanOku(const FileNode& node, const string& key, T& value) {
    const FileNode child = node[key];
    if (!child.empty()) {
        child >> value;
    }
}

void boolAlanOku(const FileNode& node, const string& key, bool& value) {
    const FileNode child = node[key];
    if (child.empty()) {
        return;
    }

    int numeric = value ? 1 : 0;
    child >> numeric;
    value = numeric != 0;
}

void pathAlanOku(const FileNode& node, const string& key, fs::path& value) {
    const FileNode child = node[key];
    if (child.empty()) {
        return;
    }

    string temp;
    child >> temp;
    if (!temp.empty()) {
        value = fs::path(temp);
    }
}

RuntimeConfig configYukle(const fs::path& config_path) {
    RuntimeConfig config;
    if (!fs::exists(config_path)) {
        return config;
    }

    FileStorage fs_config(config_path.string(), FileStorage::READ | FileStorage::FORMAT_JSON);
    if (!fs_config.isOpened()) {
        throw runtime_error("Config dosyasi acilamadi: " + config_path.string());
    }

    alanOku(fs_config.root(), "izlenecek_siniflar", config.izlenecek_siniflar);
    alanOku(fs_config.root(), "referans_siniflar", config.referans_siniflar);

    FileNode hedef_profilleri = fs_config["hedef_profilleri"];
    if (!hedef_profilleri.empty()) {
        config.hedef_profilleri.clear();
        for (FileNodeIterator it = hedef_profilleri.begin(); it != hedef_profilleri.end(); ++it) {
            const FileNode item = *it;
            TargetProfile profile;
            alanOku(item, "ad", profile.ad);
            alanOku(item, "etiket", profile.etiket);
            alanOku(item, "gorunen_ad", profile.gorunen_ad);
            alanOku(item, "gercek_genislik_cm", profile.gercek_genislik_cm);
            alanOku(item, "min_kilit_mesafe_m", profile.min_kilit_mesafe_m);
            alanOku(item, "max_kilit_mesafe_m", profile.max_kilit_mesafe_m);
            alanOku(item, "oncelik", profile.oncelik);

            if (profile.ad.empty()) {
                profile.ad = !profile.gorunen_ad.empty() ? profile.gorunen_ad : profile.etiket;
            }
            if (profile.gorunen_ad.empty()) {
                profile.gorunen_ad = !profile.ad.empty() ? profile.ad : profile.etiket;
            }

            config.hedef_profilleri.push_back(profile);
        }
    }

    FileNode distance = fs_config["distance"];
    if (!distance.empty()) {
        alanOku(distance, "hedef_gercek_genislik_cm", config.hedef_gercek_genislik_cm);
        alanOku(distance, "kamera_odak_uzakligi", config.kamera_odak_uzakligi);
    }

    FileNode detection = fs_config["detection"];
    if (!detection.empty()) {
        alanOku(detection, "detector_conf_threshold", config.detector_conf_threshold);
        alanOku(detection, "tracker_high_conf_threshold", config.tracker_high_conf_threshold);
        alanOku(detection, "tracker_low_conf_threshold", config.tracker_low_conf_threshold);
        alanOku(detection, "nms_threshold", config.nms_threshold);
    }

    FileNode tracking = fs_config["tracking"];
    if (!tracking.empty()) {
        alanOku(tracking, "track_match_iou_threshold", config.track_match_iou_threshold);
        alanOku(tracking, "low_conf_match_iou_threshold", config.low_conf_match_iou_threshold);
        alanOku(tracking, "track_max_missed_frames", config.track_max_missed_frames);
        alanOku(tracking, "track_visible_missed_frames", config.track_visible_missed_frames);
        alanOku(tracking, "track_confirm_hits", config.track_confirm_hits);
        alanOku(tracking, "primary_lock_max_lost_frames", config.primary_lock_max_lost_frames);
        alanOku(tracking, "primary_lock_track_bonus", config.primary_lock_track_bonus);
        alanOku(tracking, "primary_lock_center_weight", config.primary_lock_center_weight);
        alanOku(tracking, "primary_lock_area_weight", config.primary_lock_area_weight);
        boolAlanOku(tracking, "kilit_icin_mesafe_zorunlu", config.kilit_icin_mesafe_zorunlu);
        alanOku(tracking, "kilit_aralik_bonus", config.kilit_aralik_bonus);
    }

    FileNode runtime = fs_config["runtime"];
    if (!runtime.empty()) {
        alanOku(runtime, "camera_index", config.camera_index);
        pathAlanOku(runtime, "record_dir", config.record_dir);
        pathAlanOku(runtime, "log_dir", config.log_dir);
        pathAlanOku(runtime, "package_dir", config.package_dir);
        pathAlanOku(runtime, "worker_script", config.worker_script);
        pathAlanOku(runtime, "python_path", config.python_path);
        alanOku(runtime, "worker_jpeg_quality", config.worker_jpeg_quality);
        alanOku(runtime, "worker_max_restarts", config.worker_max_restarts);
        alanOku(runtime, "camera_reopen_attempts", config.camera_reopen_attempts);
        alanOku(runtime, "camera_reopen_delay_ms", config.camera_reopen_delay_ms);
        boolAlanOku(runtime, "show_window", config.show_window);
        boolAlanOku(runtime, "enable_recording", config.enable_recording);
        boolAlanOku(runtime, "worker_auto_restart", config.worker_auto_restart);
        boolAlanOku(runtime, "log_every_frame", config.log_every_frame);
    }

    return config;
}

void configDogrula(const RuntimeConfig& config) {
    if (config.izlenecek_siniflar.empty() && config.hedef_profilleri.empty()) {
        throw runtime_error("Config hatasi: izlenecek_siniflar bos olamaz.");
    }
    if (config.hedef_gercek_genislik_cm <= 0.0f || config.kamera_odak_uzakligi <= 0.0f) {
        throw runtime_error("Config hatasi: mesafe hesap parametreleri sifirdan buyuk olmalidir.");
    }
    if (config.detector_conf_threshold < 0.0 || config.detector_conf_threshold > 1.0 ||
        config.tracker_high_conf_threshold < 0.0 || config.tracker_high_conf_threshold > 1.0 ||
        config.tracker_low_conf_threshold < 0.0 || config.tracker_low_conf_threshold > 1.0 ||
        config.nms_threshold < 0.0 || config.nms_threshold > 1.0) {
        throw runtime_error("Config hatasi: threshold degerleri 0.0 ile 1.0 arasinda olmalidir.");
    }
    if (config.detector_conf_threshold > config.tracker_low_conf_threshold ||
        config.tracker_low_conf_threshold > config.tracker_high_conf_threshold) {
        throw runtime_error("Config hatasi: detector_conf <= tracker_low_conf <= tracker_high_conf olmalidir.");
    }
    if (config.track_match_iou_threshold <= 0.0f || config.low_conf_match_iou_threshold <= 0.0f) {
        throw runtime_error("Config hatasi: IoU esikleri sifirdan buyuk olmalidir.");
    }
    if (config.track_confirm_hits < 1 || config.track_max_missed_frames < 1 || config.track_visible_missed_frames < 0) {
        throw runtime_error("Config hatasi: tracking frame ayarlari gecersiz.");
    }
    if (config.worker_jpeg_quality < 50 || config.worker_jpeg_quality > 100) {
        throw runtime_error("Config hatasi: worker_jpeg_quality 50 ile 100 arasinda olmalidir.");
    }
    if (config.kilit_aralik_bonus < 0.0f || config.kilit_aralik_bonus > 1.0f) {
        throw runtime_error("Config hatasi: kilit_aralik_bonus 0.0 ile 1.0 arasinda olmalidir.");
    }

    set<string> profile_adlari;
    set<string> profile_etiketleri;
    for (const auto& profile : config.hedef_profilleri) {
        if (profile.ad.empty() || profile.etiket.empty()) {
            throw runtime_error("Config hatasi: hedef_profilleri icindeki her kayitta ad ve etiket dolu olmalidir.");
        }
        if (profile.gercek_genislik_cm <= 0.0f) {
            throw runtime_error("Config hatasi: hedef profil gercek_genislik_cm sifirdan buyuk olmalidir.");
        }
        if (profile.min_kilit_mesafe_m < 0.0f || profile.max_kilit_mesafe_m <= 0.0f ||
            profile.min_kilit_mesafe_m > profile.max_kilit_mesafe_m) {
            throw runtime_error("Config hatasi: hedef profil kilit mesafesi araligi gecersiz.");
        }
        if (profile.oncelik < 1) {
            throw runtime_error("Config hatasi: hedef profil oncelik en az 1 olmalidir.");
        }

        const string ad_key = kucukHarf(profile.ad);
        const string etiket_key = kucukHarf(profile.etiket);
        if (!profile_adlari.insert(ad_key).second) {
            throw runtime_error("Config hatasi: hedef profil adlari benzersiz olmalidir.");
        }
        if (!profile_etiketleri.insert(etiket_key).second) {
            throw runtime_error("Config hatasi: hedef profil etiketleri benzersiz olmalidir.");
        }
    }
}

int etiketeGoreSinifIdBul(const vector<string>& labels, const string& etiket) {
    const string hedef = kucukHarf(trim(etiket));
    for (size_t i = 0; i < labels.size(); ++i) {
        if (kucukHarf(trim(labels[i])) == hedef) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

vector<string> hedefProfilleriniEsle(RuntimeConfig& config, const vector<string>& labels) {
    vector<string> warnings;
    if (config.hedef_profilleri.empty()) {
        return warnings;
    }

    config.izlenecek_siniflar.clear();
    for (auto& profile : config.hedef_profilleri) {
        profile.resolved_class_id = etiketeGoreSinifIdBul(labels, profile.etiket);
        if (profile.resolved_class_id < 0) {
            warnings.push_back("Hedef profili eslesmedi: " + profile.ad + " -> " + profile.etiket);
            continue;
        }

        config.izlenecek_siniflar.push_back(profile.resolved_class_id);
    }

    return warnings;
}

void applyOverrides(const AppOptions& options, RuntimeConfig& config) {
    if (!options.package_override.empty()) {
        config.package_dir = options.package_override;
    }
    if (!options.python_override.empty()) {
        config.python_path = options.python_override;
    }
    if (!options.log_dir_override.empty()) {
        config.log_dir = options.log_dir_override;
    }
    if (!options.record_dir_override.empty()) {
        config.record_dir = options.record_dir_override;
    }
    if (options.camera_override >= 0) {
        config.camera_index = options.camera_override;
    }
    if (options.headless) {
        config.show_window = false;
    }
    if (options.force_show) {
        config.show_window = true;
    }
    if (options.record_override_set) {
        config.enable_recording = options.record_override_value;
    }
}

AppOptions argumanlariAyristir(int argc, char** argv) {
    AppOptions options;

    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        const auto require_value = [&](const string& name) -> string {
            if (i + 1 >= argc) {
                throw runtime_error("Eksik arguman degeri: " + name);
            }
            ++i;
            return argv[i];
        };

        if (arg == "--help" || arg == "-h") {
            cout << "Kullanim:\n"
                 << "  hss_sistem.exe [--config path] [--package dir] [--camera index]\n"
                 << "                 [--python path] [--headless|--show]\n"
                 << "                 [--record|--no-record] [--record-dir dir] [--log-dir dir]\n";
            std::exit(0);
        } else if (arg == "--config") {
            options.config_path = require_value(arg);
        } else if (arg == "--package") {
            options.package_override = require_value(arg);
        } else if (arg == "--camera") {
            options.camera_override = stoi(require_value(arg));
        } else if (arg == "--python") {
            options.python_override = require_value(arg);
        } else if (arg == "--log-dir") {
            options.log_dir_override = require_value(arg);
        } else if (arg == "--record-dir") {
            options.record_dir_override = require_value(arg);
        } else if (arg == "--headless") {
            options.headless = true;
        } else if (arg == "--show") {
            options.force_show = true;
        } else if (arg == "--record") {
            options.record_override_set = true;
            options.record_override_value = true;
        } else if (arg == "--no-record") {
            options.record_override_set = true;
            options.record_override_value = false;
        } else if (!basliyorMu(arg, "--") && options.package_override.empty()) {
            options.package_override = arg;
        } else {
            throw runtime_error("Bilinmeyen arguman: " + arg);
        }
    }

    return options;
}

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

wstring pythonYoluBul(const fs::path& configured_python) {
    vector<fs::path> candidates;

    if (!configured_python.empty()) {
        candidates.push_back(configured_python);
    }

    char env_buffer[1024] = {};
    const DWORD env_length = GetEnvironmentVariableA("KORGAN_PYTHON", env_buffer, static_cast<DWORD>(sizeof(env_buffer)));
    if (env_length > 0 && env_length < sizeof(env_buffer)) {
        candidates.emplace_back(env_buffer);
    }

    candidates.emplace_back("C:/Users/askan/AppData/Local/Programs/Python/Python312/python.exe");
    candidates.emplace_back("python.exe");

    for (const auto& candidate : candidates) {
        if (candidate.empty()) {
            continue;
        }
        if (candidate.filename() == "python.exe" || fs::exists(candidate)) {
            return candidate.wstring();
        }
    }

    throw runtime_error("Python bulunamadi. KORGAN_PYTHON veya --python ile tam yolu verin.");
}

class JsonlLogger {
public:
    explicit JsonlLogger(const fs::path& log_dir) {
        fs::create_directories(log_dir);
        path_ = fs::absolute(log_dir / ("telemetry_" + dosyaAdiIcinZamanDamgasi() + ".jsonl"));
        output_.open(path_, ios::out | ios::app);
        if (!output_.is_open()) {
            throw runtime_error("Log dosyasi acilamadi: " + path_.string());
        }
    }

    const fs::path& path() const {
        return path_;
    }

    void yaz(const string& line) {
        output_ << line << '\n';
        output_.flush();
    }

private:
    fs::path path_;
    ofstream output_;
};

class FrameRecorder {
public:
    void baslat(const fs::path& record_dir, const Size& frame_size, double fps) {
        if (writer_.isOpened()) {
            return;
        }

        fs::create_directories(record_dir);
        output_path_ = fs::absolute(record_dir / ("session_" + dosyaAdiIcinZamanDamgasi() + ".mp4"));

        const double normalized_fps = (fps > 1.0 && fps < 240.0) ? fps : 30.0;
        writer_.open(output_path_.string(), VideoWriter::fourcc('m', 'p', '4', 'v'), normalized_fps, frame_size, true);

        if (!writer_.isOpened()) {
            output_path_ = fs::absolute(record_dir / ("session_" + dosyaAdiIcinZamanDamgasi() + ".avi"));
            writer_.open(output_path_.string(), VideoWriter::fourcc('M', 'J', 'P', 'G'), normalized_fps, frame_size, true);
        }

        if (!writer_.isOpened()) {
            throw runtime_error("Kayit dosyasi acilamadi.");
        }
    }

    bool aktif() const {
        return writer_.isOpened();
    }

    const fs::path& yol() const {
        return output_path_;
    }

    void kareYaz(const Mat& frame) {
        if (writer_.isOpened()) {
            writer_.write(frame);
        }
    }

private:
    VideoWriter writer_;
    fs::path output_path_;
};

class DetectionTracker {
public:
    explicit DetectionTracker(const RuntimeConfig& config) : config_(config) {}

    vector<Detection> guncelle(const vector<Detection>& raw_detections, const Size& frame_size) {
        vector<Detection> detections = raw_detections;
        vector<Rect> predicted_boxes(tracks_.size());
        vector<size_t> active_track_indices;

        for (size_t i = 0; i < tracks_.size(); ++i) {
            Mat prediction = tracks_[i].filter.predict();
            tracks_[i].estimated_box = stateTenKutuOlustur(prediction, frame_size);
            predicted_boxes[i] = tracks_[i].estimated_box;
            tracks_[i].missed_frames++;
            active_track_indices.push_back(i);
        }

        vector<size_t> high_indices;
        vector<size_t> low_indices;
        for (size_t i = 0; i < detections.size(); ++i) {
            detections[i].box = kutuyuKirp(detections[i].box, frame_size);
            if (detections[i].box.width <= 0 || detections[i].box.height <= 0) {
                continue;
            }

            if (detections[i].confidence >= config_.tracker_high_conf_threshold) {
                high_indices.push_back(i);
            } else if (detections[i].confidence >= config_.tracker_low_conf_threshold) {
                low_indices.push_back(i);
            }
        }

        vector<bool> detection_used(detections.size(), false);
        vector<bool> track_used(tracks_.size(), false);
        vector<Detection> output;
        output.reserve(detections.size() + tracks_.size());

        eslestir(high_indices,
                 config_.track_match_iou_threshold,
                 false,
                 detections,
                 predicted_boxes,
                 detection_used,
                 track_used,
                 output,
                 frame_size);

        vector<size_t> unmatched_confirmed_tracks;
        for (size_t i = 0; i < tracks_.size(); ++i) {
            if (!track_used[i] && tracks_[i].confirmed) {
                unmatched_confirmed_tracks.push_back(i);
            }
        }

        eslestir(low_indices,
                 config_.low_conf_match_iou_threshold,
                 true,
                 detections,
                 predicted_boxes,
                 detection_used,
                 track_used,
                 output,
                 frame_size,
                 unmatched_confirmed_tracks);

        for (size_t det_index : high_indices) {
            if (detection_used[det_index]) {
                continue;
            }

            Track new_track;
            new_track.id = next_track_id_++;
            new_track.class_id = detections[det_index].class_id;
            new_track.hits = 1;
            new_track.confirmed = (config_.track_confirm_hits <= 1);
            new_track.last_confidence = detections[det_index].confidence;
            new_track.filter = kalmanFiltresiOlustur(detections[det_index].box);
            new_track.estimated_box = kutuyuKirp(detections[det_index].box, frame_size);
            tracks_.push_back(new_track);

            Detection detected = detections[det_index];
            detected.box = new_track.estimated_box;
            detected.track_id = new_track.id;
            detected.kalman_kullanildi = true;
            output.push_back(detected);
        }

        for (Track& track : tracks_) {
            if (!track.confirmed || track.missed_frames <= 0 || track.missed_frames > config_.track_visible_missed_frames) {
                continue;
            }

            Detection predicted;
            predicted.class_id = track.class_id;
            predicted.confidence = std::max(track.last_confidence * 0.85f, 0.05f);
            predicted.box = kutuyuKirp(track.estimated_box, frame_size);
            predicted.track_id = track.id;
            predicted.kalman_kullanildi = true;
            predicted.predicted_only = true;
            output.push_back(predicted);
        }

        tracks_.erase(remove_if(tracks_.begin(),
                                tracks_.end(),
                                [&](const Track& track) {
                                    return track.missed_frames > config_.track_max_missed_frames;
                                }),
                      tracks_.end());

        return output;
    }

private:
    RuntimeConfig config_;
    int next_track_id_ = 1;
    vector<Track> tracks_;

    void eslestir(const vector<size_t>& detection_indices,
                  float min_iou,
                  bool low_confidence_pass,
                  vector<Detection>& detections,
                  const vector<Rect>& predicted_boxes,
                  vector<bool>& detection_used,
                  vector<bool>& track_used,
                  vector<Detection>& output,
                  const Size& frame_size,
                  const vector<size_t>& track_filter = {}) {
        vector<size_t> candidate_tracks = track_filter;
        if (candidate_tracks.empty()) {
            candidate_tracks.resize(tracks_.size());
            for (size_t i = 0; i < tracks_.size(); ++i) {
                candidate_tracks[i] = i;
            }
        }

        vector<MatchCandidate> candidates;
        for (size_t track_index : candidate_tracks) {
            if (track_index >= tracks_.size() || track_used[track_index]) {
                continue;
            }

            for (size_t det_index : detection_indices) {
                if (det_index >= detections.size() || detection_used[det_index]) {
                    continue;
                }
                if (tracks_[track_index].class_id != detections[det_index].class_id) {
                    continue;
                }

                const float iou = kutuIou(predicted_boxes[track_index], detections[det_index].box);
                if (iou < min_iou) {
                    continue;
                }

                const float score = iou + detections[det_index].confidence * 0.01f;
                candidates.push_back({track_index, det_index, score});
            }
        }

        sort(candidates.begin(), candidates.end(), [](const MatchCandidate& a, const MatchCandidate& b) {
            return a.score > b.score;
        });

        for (const MatchCandidate& candidate : candidates) {
            if (track_used[candidate.track_index] || detection_used[candidate.detection_index]) {
                continue;
            }

            Track& track = tracks_[candidate.track_index];
            Detection& detection = detections[candidate.detection_index];
            const Point2f center = kutuMerkezi(detection.box);

            Mat measurement(4, 1, CV_32F);
            measurement.at<float>(0) = center.x;
            measurement.at<float>(1) = center.y;
            measurement.at<float>(2) = static_cast<float>(std::max(detection.box.width, 1));
            measurement.at<float>(3) = static_cast<float>(std::max(detection.box.height, 1));

            Mat estimated = track.filter.correct(measurement);
            track.estimated_box = stateTenKutuOlustur(estimated, frame_size);
            track.last_confidence = detection.confidence;
            track.missed_frames = 0;
            track.hits++;
            if (track.hits >= config_.track_confirm_hits) {
                track.confirmed = true;
            }

            detection.box = track.estimated_box;
            detection.track_id = track.id;
            detection.kalman_kullanildi = true;
            detection.low_confidence_match = low_confidence_pass;

            output.push_back(detection);
            detection_used[candidate.detection_index] = true;
            track_used[candidate.track_index] = true;
        }
    }
};

class OrtWorkerClient {
public:
    explicit OrtWorkerClient(const WorkerLaunchOptions& options) : options_(options) {
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

        const fs::path worker_path = fs::absolute(options_.worker_script);
        if (!fs::exists(worker_path)) {
            CloseHandle(child_stdout_read);
            CloseHandle(child_stdout_write);
            CloseHandle(child_stdin_read);
            CloseHandle(child_stdin_write);
            throw runtime_error("Worker script bulunamadi: " + worker_path.string());
        }

        const wstring python_path = pythonYoluBul(options_.python_path);
        wostringstream command_stream;
        command_stream << L"\"" << python_path << L"\" "
                       << L"\"" << worker_path.wstring() << L"\" "
                       << L"--package \"" << fs::absolute(options_.package_dir).wstring() << L"\" "
                       << L"--conf-threshold " << options_.detector_conf_threshold << L" "
                       << L"--nms-threshold " << options_.nms_threshold;
        const wstring command = command_stream.str();

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
        if (!calisiyorMu()) {
            throw runtime_error("Worker sureci calismiyor.");
        }

        vector<uchar> jpeg_buffer;
        vector<int> jpeg_params = {IMWRITE_JPEG_QUALITY, options_.jpeg_quality};
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
    WorkerLaunchOptions options_;
    HANDLE stdout_read_ = nullptr;
    HANDLE stdin_write_ = nullptr;
    PROCESS_INFORMATION process_info_{};

    bool calisiyorMu() const {
        if (process_info_.hProcess == nullptr) {
            return false;
        }
        return WaitForSingleObject(process_info_.hProcess, 0) == WAIT_TIMEOUT;
    }

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
        FileStorage fs_json(response, FileStorage::READ | FileStorage::MEMORY | FileStorage::FORMAT_JSON);
        if (!fs_json.isOpened()) {
            throw runtime_error("Worker cevabi JSON olarak okunamadi: " + response);
        }

        string error_message;
        fs_json["error"] >> error_message;
        if (!error_message.empty()) {
            throw runtime_error("Worker hata verdi: " + error_message);
        }

        InferenceResult result;
        FileNode timings_node = fs_json["timings_ms"];
        if (!timings_node.empty()) {
            timings_node["decode"] >> result.timings.decode_ms;
            timings_node["preprocess"] >> result.timings.preprocess_ms;
            timings_node["inference"] >> result.timings.inference_ms;
            timings_node["postprocess"] >> result.timings.postprocess_ms;
            timings_node["worker_total"] >> result.timings.worker_total_ms;
        }

        FileNode detections_node = fs_json["detections"];
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

            det.box = Rect(Point(bbox[0], bbox[1]), Point(bbox[2], bbox[3]));
            result.detections.push_back(det);
        }

        return result;
    }
};

bool kameraAc(VideoCapture& cap, int camera_index, int attempts, int delay_ms) {
    for (int attempt = 0; attempt < attempts; ++attempt) {
        cap.release();
        if (cap.open(camera_index)) {
            return true;
        }
        this_thread::sleep_for(chrono::milliseconds(delay_ms));
    }
    return false;
}

string detectionJson(const Detection& detection, const vector<string>& labels) {
    const string label = (detection.class_id >= 0 && detection.class_id < static_cast<int>(labels.size()))
                             ? labels[detection.class_id]
                             : ("class_" + to_string(detection.class_id));

    ostringstream out;
    out << "{"
        << "\"class_id\":" << detection.class_id << ','
        << "\"label\":\"" << jsonKacis(label) << "\","
        << "\"track_id\":" << detection.track_id << ','
        << "\"confidence\":" << fixed << setprecision(4) << detection.confidence << ','
        << "\"predicted_only\":" << jsonBool(detection.predicted_only) << ','
        << "\"low_confidence_match\":" << jsonBool(detection.low_confidence_match) << ','
        << "\"bbox_xywh\":[" << detection.box.x << ',' << detection.box.y << ',' << detection.box.width << ','
        << detection.box.height << "]}";
    return out.str();
}

void telemetrySatiriYaz(JsonlLogger& logger,
                        int frame_index,
                        TakipDurumu takip_durumu,
                        const vector<Detection>& detections,
                        const vector<string>& labels,
                        const TimingMetrics& timings,
                        double worker_round_trip_ms,
                        double tracker_ms,
                        double render_ms,
                        double frame_total_ms,
                        double fps,
                        int worker_restart_count,
                        bool recorder_active,
                        const string& recorder_path,
                        bool primary_visible,
                        int primary_target_id,
                        int primary_track_id,
                        const string& primary_profile_name,
                        const string& primary_label,
                        const string& primary_confidence,
                        int primary_cx,
                        int primary_cy,
                        int primary_z_cm,
                        bool primary_in_lock_window,
                        double primary_min_lock_distance_m,
                        double primary_max_lock_distance_m,
                        bool primary_engagement_allowed,
                        int primary_lost_frames) {
    ostringstream out;
    out << "{"
        << "\"ts\":\"" << jsonKacis(zamanDamgasiOlustur()) << "\","
        << "\"event\":\"frame\","
        << "\"frame_index\":" << frame_index << ','
        << "\"tracking_status\":\"" << takipDurumuMetni(takip_durumu) << "\","
        << "\"worker_restart_count\":" << worker_restart_count << ','
        << "\"recorder_active\":" << jsonBool(recorder_active) << ','
        << "\"recorder_path\":\"" << jsonKacis(recorder_path) << "\","
        << "\"fps\":" << fixed << setprecision(2) << fps << ','
        << "\"timings_ms\":{"
        << "\"round_trip\":" << worker_round_trip_ms << ','
        << "\"decode\":" << timings.decode_ms << ','
        << "\"preprocess\":" << timings.preprocess_ms << ','
        << "\"inference\":" << timings.inference_ms << ','
        << "\"postprocess\":" << timings.postprocess_ms << ','
        << "\"worker_total\":" << timings.worker_total_ms << ','
        << "\"tracker\":" << tracker_ms << ','
        << "\"render\":" << render_ms << ','
        << "\"frame_total\":" << frame_total_ms << "},";

    if (primary_visible) {
        out << "\"primary_target\":{"
            << "\"class_id\":" << primary_target_id << ','
            << "\"profile\":\"" << jsonKacis(primary_profile_name) << "\","
            << "\"label\":\"" << jsonKacis(primary_label) << "\","
            << "\"track_id\":" << primary_track_id << ','
            << "\"confidence\":\"" << jsonKacis(primary_confidence) << "\","
            << "\"x\":" << primary_cx << ','
            << "\"y\":" << primary_cy << ','
            << "\"z_cm\":" << primary_z_cm << ','
            << "\"distance_m\":" << fixed << setprecision(2) << (primary_z_cm / 100.0) << ','
            << "\"in_lock_window\":" << jsonBool(primary_in_lock_window) << ','
            << "\"lock_window_m\":[" << fixed << setprecision(2) << primary_min_lock_distance_m << ','
            << primary_max_lock_distance_m << "],"
            << "\"engagement_allowed\":" << jsonBool(primary_engagement_allowed) << "},";
    } else {
        out << "\"primary_target\":null,";
    }

    out << "\"primary_lost_frames\":" << primary_lost_frames << ','
        << "\"detections\":[";

    for (size_t i = 0; i < detections.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << detectionJson(detections[i], labels);
    }

    out << "]}";
    logger.yaz(out.str());
}

int main(int argc, char** argv) {
    try {
        const AppOptions options = argumanlariAyristir(argc, argv);
        RuntimeConfig config = configYukle(options.config_path);
        applyOverrides(options, config);
        configDogrula(config);

        const PackageInfo package_info = paketiYukle(config.package_dir);
        const vector<string> labels = etiketleriYukle(package_info.labels_path);
        if (package_info.class_count > 0 && static_cast<int>(labels.size()) != package_info.class_count) {
            throw runtime_error("labels.txt sinif sayisi manifest ile uyusmuyor.");
        }

        if (!fs::exists(config.worker_script)) {
            throw runtime_error("Worker script bulunamadi: " + config.worker_script.string());
        }

        cout << "==========================================" << endl;
        cout << "[SISTEM] Profesyonel runtime baslatiliyor..." << endl;
        cout << "[CONFIG] " << fs::absolute(options.config_path).string() << endl;
        cout << "[PAKET] " << package_info.package_dir.string() << endl;
        cout << "[MODEL] " << package_info.model_path.filename().string() << endl;
        cout << "[ETIKET] " << labels.size() << " sinif" << endl;
        cout << "==========================================" << endl;

        JsonlLogger logger(config.log_dir);
        logger.yaz("{\"ts\":\"" + jsonKacis(zamanDamgasiOlustur()) + "\",\"event\":\"startup\",\"log_path\":\"" +
                   jsonKacis(logger.path().string()) + "\"}");

        const vector<string> target_profile_warnings = hedefProfilleriniEsle(config, labels);
        for (const auto& warning : target_profile_warnings) {
            cout << "[UYARI] " << warning << endl;
            logger.yaz("{\"ts\":\"" + jsonKacis(zamanDamgasiOlustur()) + "\",\"event\":\"target_profile_warning\",\"message\":\"" +
                       jsonKacis(warning) + "\"}");
        }

        if (!config.hedef_profilleri.empty()) {
            cout << "[HEDEFLER] " << config.izlenecek_siniflar.size() << "/" << config.hedef_profilleri.size()
                 << " profil aktif." << endl;
        }

        WorkerLaunchOptions worker_options;
        worker_options.package_dir = package_info.package_dir;
        worker_options.worker_script = config.worker_script;
        worker_options.python_path = config.python_path;
        worker_options.detector_conf_threshold = config.detector_conf_threshold;
        worker_options.nms_threshold = config.nms_threshold;
        worker_options.jpeg_quality = config.worker_jpeg_quality;

        auto worker = make_unique<OrtWorkerClient>(worker_options);
        int worker_restart_count = 0;

        DetectionTracker tracker(config);
        PrimaryLockState primary_lock;
        FrameRecorder recorder;

        VideoCapture cap;
        if (!kameraAc(cap, config.camera_index, config.camera_reopen_attempts, config.camera_reopen_delay_ms)) {
            throw runtime_error("Kamera acilamadi.");
        }

        logger.yaz("{\"ts\":\"" + jsonKacis(zamanDamgasiOlustur()) + "\",\"event\":\"camera_ready\",\"camera_index\":" +
                   to_string(config.camera_index) + "}");

        Mat frame;
        double smoothed_fps = 0.0;
        size_t previous_terminal_status_length = 0;
        int frame_index = 0;

        while (true) {
            const auto frame_start = chrono::steady_clock::now();

            if (!cap.read(frame) || frame.empty()) {
                logger.yaz("{\"ts\":\"" + jsonKacis(zamanDamgasiOlustur()) +
                           "\",\"event\":\"camera_read_failed\",\"action\":\"reopen\"}");
                if (!kameraAc(cap, config.camera_index, config.camera_reopen_attempts, config.camera_reopen_delay_ms) ||
                    !cap.read(frame) || frame.empty()) {
                    throw runtime_error("Kamera frame okuyamiyor.");
                }
            }

            if (config.enable_recording && !recorder.aktif()) {
                const double camera_fps = cap.get(CAP_PROP_FPS);
                recorder.baslat(config.record_dir, frame.size(), camera_fps);
                logger.yaz("{\"ts\":\"" + jsonKacis(zamanDamgasiOlustur()) + "\",\"event\":\"recording_started\",\"path\":\"" +
                           jsonKacis(recorder.yol().string()) + "\"}");
            }

            const auto worker_start = chrono::steady_clock::now();
            InferenceResult inference_result;
            try {
                inference_result = worker->tahminEt(frame);
            } catch (const exception& ex) {
                logger.yaz("{\"ts\":\"" + jsonKacis(zamanDamgasiOlustur()) + "\",\"event\":\"worker_error\",\"message\":\"" +
                           jsonKacis(ex.what()) + "\"}");
                if (!config.worker_auto_restart || worker_restart_count >= config.worker_max_restarts) {
                    throw;
                }

                worker = make_unique<OrtWorkerClient>(worker_options);
                worker_restart_count++;
                logger.yaz("{\"ts\":\"" + jsonKacis(zamanDamgasiOlustur()) +
                           "\",\"event\":\"worker_restarted\",\"restart_count\":" + to_string(worker_restart_count) + "}");
                inference_result = worker->tahminEt(frame);
            }

            const auto worker_end = chrono::steady_clock::now();
            const vector<Detection> detections = tracker.guncelle(inference_result.detections, frame.size());
            const auto tracker_end = chrono::steady_clock::now();
            const int primary_index = birincilHedefiSec(config, detections, frame.size(), primary_lock);

            if (primary_index >= 0) {
                primary_lock.active_track_id = detections[primary_index].track_id;
                primary_lock.lost_frames = 0;
            } else if (primary_lock.active_track_id >= 0) {
                primary_lock.lost_frames++;
                if (primary_lock.lost_frames > config.primary_lock_max_lost_frames) {
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
            string primary_profile_name = "-";
            string primary_label = "-";
            string primary_confidence = "0.00";
            float primary_confidence_value = 0.0f;
            bool primary_in_lock_window = false;
            bool primary_engagement_allowed = false;
            double primary_min_lock_distance_m = 0.0;
            double primary_max_lock_distance_m = 0.0;

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
                const TargetProfile* target_profile = hedefProfiliBul(config, nesne_id);
                const string gorunen_label =
                    (target_profile != nullptr && !target_profile->gorunen_ad.empty()) ? target_profile->gorunen_ad : label;

                const int cx = box.x + box.width / 2;
                const int cy = box.y + box.height / 2;
                const int mesafe_z_cm = hedefMesafesiCmHesapla(config, detection, target_profile);
                const bool lock_in_range = hedefKilitMesafesindeMi(target_profile, mesafe_z_cm);

                string takip_text = "T" + to_string(detection.track_id);
                if (detection.predicted_only) {
                    takip_text += " PRED";
                } else if (detection.low_confidence_match) {
                    takip_text += " LOW";
                } else if (detection.kalman_kullanildi) {
                    takip_text += " KF";
                }

                const string confidence_text = format("%.2f", detection.confidence);
                const bool primary_lock_active =
                    primary_index >= 0 && static_cast<int>(detection_index) == primary_index &&
                    detection.track_id == primary_lock.active_track_id;

                if (listedeVarMi(config.referans_siniflar, nesne_id)) {
                    const Scalar ref_color = detection.predicted_only ? Scalar(180, 0, 0) : Scalar(255, 0, 0);
                    rectangle(frame, box, ref_color, 2);
                    putText(frame,
                            "REFERANS " + label + " " + to_string(mesafe_z_cm) + " cm",
                            Point(box.x, std::max(box.y - 10, 20)),
                            FONT_HERSHEY_SIMPLEX,
                            0.55,
                            ref_color,
                            2);
                    putText(frame,
                            takip_text,
                            Point(box.x, std::min(box.y + box.height + 18, frame.rows - 10)),
                            FONT_HERSHEY_SIMPLEX,
                            0.45,
                            Scalar(255, 180, 0),
                            1);
                } else if (listedeVarMi(config.izlenecek_siniflar, nesne_id)) {
                    const Scalar tracked_color = primary_lock_active
                                                     ? (lock_in_range ? Scalar(0, 255, 0) : Scalar(0, 170, 255))
                                                     : (detection.predicted_only ? Scalar(80, 80, 255)
                                                                                 : (lock_in_range ? Scalar(0, 220, 255)
                                                                                                  : Scalar(0, 140, 255)));
                    const Scalar center_color = lock_in_range ? Scalar(0, 255, 0) : Scalar(0, 220, 255);
                    rectangle(frame, box, tracked_color, primary_lock_active ? 2 : 1);
                    circle(frame, Point(cx, cy), primary_lock_active ? 5 : 3, center_color, -1);

                    if (primary_lock_active) {
                        primary_telemetry_visible = true;
                        primary_cx = cx;
                        primary_cy = cy;
                        primary_z_cm = mesafe_z_cm;
                        primary_target_id = nesne_id;
                        primary_track_id = detection.track_id;
                        primary_profile_name = target_profile != nullptr ? target_profile->ad : label;
                        primary_label = gorunen_label;
                        primary_confidence = confidence_text;
                        primary_confidence_value = detection.confidence;
                        primary_in_lock_window = lock_in_range;
                        primary_engagement_allowed = lock_in_range;
                        primary_min_lock_distance_m =
                            target_profile != nullptr ? target_profile->min_kilit_mesafe_m : 0.0;
                        primary_max_lock_distance_m =
                            target_profile != nullptr ? target_profile->max_kilit_mesafe_m : 0.0;

                        putText(frame,
                                string(lock_in_range ? "KILIT HAZIR " : "ODAK ") + gorunen_label + " @" +
                                    confidence_text + " Z:" + format("%.2f", mesafe_z_cm / 100.0) + "m",
                                Point(box.x, std::max(box.y - 10, 20)),
                                FONT_HERSHEY_SIMPLEX,
                                0.55,
                                lock_in_range ? Scalar(0, 255, 0) : Scalar(0, 170, 255),
                                2);
                        putText(frame,
                                "X:" + to_string(cx) + " Y:" + to_string(cy) + " Z:" +
                                    format("%.2f", mesafe_z_cm / 100.0) + "m" +
                                    (target_profile != nullptr
                                         ? " ARALIK:" + format("%.1f", target_profile->min_kilit_mesafe_m) + "-" +
                                               format("%.1f", target_profile->max_kilit_mesafe_m) + "m"
                                         : ""),
                                Point(box.x, std::min(box.y + box.height + 20, frame.rows - 10)),
                                FONT_HERSHEY_SIMPLEX,
                                0.5,
                                lock_in_range ? Scalar(0, 255, 0) : Scalar(0, 220, 255),
                                2);
                        putText(frame,
                                takip_text + (lock_in_range ? " ATIS_PENCERESI" : " MESAFE_DISI"),
                                Point(std::max(box.x - 2, 0), std::min(box.y + box.height + 40, frame.rows - 10)),
                                FONT_HERSHEY_SIMPLEX,
                                0.45,
                                Scalar(255, 220, 0),
                                1);
                    } else {
                        putText(frame,
                                (detection.predicted_only ? "TAHMIN " : "IZLEME ") + gorunen_label + " @" +
                                    confidence_text + " " + takip_text +
                                    (target_profile != nullptr
                                         ? " Z:" + format("%.2f", mesafe_z_cm / 100.0) + "m"
                                         : ""),
                                Point(box.x, std::max(box.y - 8, 20)),
                                FONT_HERSHEY_SIMPLEX,
                                0.45,
                                tracked_color,
                                1);
                    }
                } else {
                    const Scalar passive_color = detection.predicted_only ? Scalar(100, 100, 100) : Scalar(0, 255, 255);
                    rectangle(frame, box, passive_color, 1);
                    putText(frame,
                            label + " @" + confidence_text + " " + takip_text,
                            Point(box.x, std::max(box.y - 8, 20)),
                            FONT_HERSHEY_SIMPLEX,
                            0.45,
                            passive_color,
                            1);
                }
            }

            putText(frame,
                    "BYTE-STYLE TRACK  det>" + format("%.2f", config.detector_conf_threshold) + "  hi>" +
                        format("%.2f", config.tracker_high_conf_threshold) + "  lo>" +
                        format("%.2f", config.tracker_low_conf_threshold),
                    Point(20, 30),
                    FONT_HERSHEY_SIMPLEX,
                    0.55,
                    Scalar(255, 255, 255),
                    2);

            putText(frame,
                    primary_lock.active_track_id >= 0 ? "AKTIF TAKIP: T" + to_string(primary_lock.active_track_id)
                                                      : "AKTIF TAKIP: BEKLENIYOR",
                    Point(20, 58),
                    FONT_HERSHEY_SIMPLEX,
                    0.55,
                    primary_lock.active_track_id >= 0 ? Scalar(0, 255, 0) : Scalar(0, 220, 255),
                    2);

            if (recorder.aktif()) {
                putText(frame,
                        "REC",
                        Point(20, 86),
                        FONT_HERSHEY_SIMPLEX,
                        0.65,
                        Scalar(0, 0, 255),
                        2);
            }

            if (worker_restart_count > 0) {
                putText(frame,
                        "WORKER RESTART:" + to_string(worker_restart_count),
                        Point(90, 86),
                        FONT_HERSHEY_SIMPLEX,
                        0.5,
                        Scalar(0, 165, 255),
                        1);
            }

            const auto overlay_end = chrono::steady_clock::now();
            const double worker_round_trip_ms = chrono::duration<double, milli>(worker_end - worker_start).count();
            const double tracker_ms = chrono::duration<double, milli>(tracker_end - worker_end).count();
            const double render_ms = chrono::duration<double, milli>(overlay_end - tracker_end).count();
            const double frame_total_ms = chrono::duration<double, milli>(overlay_end - frame_start).count();
            const double instantaneous_fps = frame_total_ms > 0.0 ? 1000.0 / frame_total_ms : 0.0;
            smoothed_fps = (smoothed_fps <= 0.0) ? instantaneous_fps : (smoothed_fps * 0.9 + instantaneous_fps * 0.1);

            const TakipDurumu takip_durumu = primary_telemetry_visible
                                                 ? (primary_in_lock_window ? TakipDurumu::KilitHazir
                                                                           : TakipDurumu::Izleniyor)
                                                 : (primary_lock.active_track_id >= 0 ? TakipDurumu::Belirsiz
                                                                                      : TakipDurumu::Bekleniyor);
            const double track_health =
                primary_telemetry_visible
                    ? std::min(100.0, std::max(0.0, static_cast<double>(primary_confidence_value) * 100.0))
                    : std::max(0.0,
                               100.0 - (primary_lock.lost_frames * 100.0 /
                                         std::max(config.primary_lock_max_lost_frames, 1)));

            const string telemetry_line =
                primary_telemetry_visible
                    ? "TS:" + zamanDamgasiOlustur() + " DURUM:" + takipDurumuMetni(takip_durumu) +
                          " X:" + to_string(primary_cx) + " Y:" + to_string(primary_cy) +
                          " Z:" + format("%.2f", primary_z_cm / 100.0) + "m ID:" + to_string(primary_target_id) + " " +
                          primary_label + " T:" + to_string(primary_track_id) + " CONF:" + primary_confidence +
                          " KILIT:" + (primary_in_lock_window ? "UYGUN" : "MESAFE_DISI") +
                          " HEALTH:" + format("%.0f", track_health) + "%"
                    : "TS:" + zamanDamgasiOlustur() + " DURUM:" + takipDurumuMetni(takip_durumu) +
                          " T:" + (primary_lock.active_track_id >= 0 ? to_string(primary_lock.active_track_id) : "-") +
                          " KAYIP:" + to_string(primary_lock.lost_frames) + " HEALTH:" +
                          format("%.0f", track_health) + "%";

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
                previous_terminal_status_length > terminal_status.size()
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

            if (config.log_every_frame) {
                telemetrySatiriYaz(logger,
                                   frame_index,
                                   takip_durumu,
                                   detections,
                                   labels,
                                   inference_result.timings,
                                   worker_round_trip_ms,
                                   tracker_ms,
                                   render_ms,
                                   frame_total_ms,
                                   smoothed_fps,
                                   worker_restart_count,
                                   recorder.aktif(),
                                   recorder.aktif() ? recorder.yol().string() : "",
                                   primary_telemetry_visible,
                                   primary_target_id,
                                   primary_track_id,
                                   primary_profile_name,
                                   primary_label,
                                   primary_confidence,
                                   primary_cx,
                                   primary_cy,
                                   primary_z_cm,
                                   primary_in_lock_window,
                                   primary_min_lock_distance_m,
                                   primary_max_lock_distance_m,
                                   primary_engagement_allowed,
                                   primary_lock.lost_frames);
            }

            if (recorder.aktif()) {
                recorder.kareYaz(frame);
            }

            if (config.show_window) {
                imshow("KORGAN - Professional Runtime", frame);
                if (waitKey(1) == 27) {
                    break;
                }
            }

            frame_index++;
        }

        if (previous_terminal_status_length > 0) {
            cout << endl;
        }

        logger.yaz("{\"ts\":\"" + jsonKacis(zamanDamgasiOlustur()) + "\",\"event\":\"shutdown\"}");
        cap.release();
        destroyAllWindows();
        return 0;
    } catch (const exception& ex) {
        cerr << "[HATA] " << ex.what() << endl;
        return 1;
    }
}
