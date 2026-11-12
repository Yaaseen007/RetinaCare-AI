import 'package:shared_preferences/shared_preferences.dart';
import '../models/prediction_result.dart';
import '../core/constants/api_constants.dart';

/// RetinaCare AI - Local Storage Service (History & Settings)
class StorageService {
  static const String _keyHistory = 'retinacare_scan_history';
  static const String _keyBaseUrl = 'retinacare_api_base_url';
  static const String _keyClinicianName = 'retinacare_clinician_name';
  static const String _keyClinicFacility = 'retinacare_clinic_facility';

  static SharedPreferences? _prefs;

  static Future<SharedPreferences> get _instance async {
    _prefs ??= await SharedPreferences.getInstance();
    return _prefs!;
  }

  // --------------------------------------------------------------------------
  // API Configuration Persistence
  // --------------------------------------------------------------------------
  static Future<String> getBaseUrl() async {
    final prefs = await _instance;
    return prefs.getString(_keyBaseUrl) ?? ApiConstants.defaultBaseUrl;
  }

  static Future<void> setBaseUrl(String url) async {
    final prefs = await _instance;
    await prefs.setString(_keyBaseUrl, url.trim());
  }

  static Future<void> resetBaseUrl() async {
    final prefs = await _instance;
    await prefs.remove(_keyBaseUrl);
  }

  // --------------------------------------------------------------------------
  // Clinician Profile Info
  // --------------------------------------------------------------------------
  static Future<String> getClinicianName() async {
    final prefs = await _instance;
    return prefs.getString(_keyClinicianName) ?? 'Dr. Retinal Specialist';
  }

  static Future<void> setClinicianName(String name) async {
    final prefs = await _instance;
    await prefs.setString(_keyClinicianName, name.trim());
  }

  static Future<String> getClinicFacility() async {
    final prefs = await _instance;
    return prefs.getString(_keyClinicFacility) ?? 'GM University Ophthalmology Center';
  }

  static Future<void> setClinicFacility(String facility) async {
    final prefs = await _instance;
    await prefs.setString(_keyClinicFacility, facility.trim());
  }

  // --------------------------------------------------------------------------
  // Scan History Persistence
  // --------------------------------------------------------------------------
  static Future<List<PredictionResult>> getScanHistory() async {
    final prefs = await _instance;
    final List<String> rawList = prefs.getStringList(_keyHistory) ?? [];
    final List<PredictionResult> results = [];

    for (final item in rawList) {
      try {
        results.add(PredictionResult.fromJsonString(item));
      } catch (_) {
        // Skip corrupted entries
      }
    }
    // Return newest first
    return results.reversed.toList();
  }

  static Future<void> saveScan(PredictionResult scan) async {
    final prefs = await _instance;
    final List<String> current = prefs.getStringList(_keyHistory) ?? [];
    current.add(scan.toJsonString());
    // Keep up to 100 historical scans
    if (current.length > 100) {
      current.removeAt(0);
    }
    await prefs.setStringList(_keyHistory, current);
  }

  static Future<void> deleteScan(String id) async {
    final prefs = await _instance;
    final List<String> current = prefs.getStringList(_keyHistory) ?? [];
    current.removeWhere((item) {
      try {
        final parsed = PredictionResult.fromJsonString(item);
        return parsed.id == id;
      } catch (_) {
        return false;
      }
    });
    await prefs.setStringList(_keyHistory, current);
  }

  static Future<void> clearAllHistory() async {
    final prefs = await _instance;
    await prefs.remove(_keyHistory);
  }
}

