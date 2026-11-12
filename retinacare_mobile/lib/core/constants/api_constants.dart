import 'package:flutter/foundation.dart';

/// RetinaCare AI - API & Clinical Constants
class ApiConstants {
  ApiConstants._();

  /// Default API base URL based on runtime platform
  /// - Android Emulator requires 10.0.2.2 to reach host machine
  /// - Web / Desktop uses 127.0.0.1 (or localhost)
  /// - Physical Android device uses local network IP (configured in Profile/Settings)
  static String get defaultBaseUrl {
    if (kIsWeb) {
      return 'http://127.0.0.1:8000';
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return 'http://10.0.2.2:8000';
      case TargetPlatform.iOS:
      case TargetPlatform.macOS:
      case TargetPlatform.windows:
      case TargetPlatform.linux:
      default:
        return 'http://127.0.0.1:8000';
    }
  }

  // Dedicated Mobile Prediction Endpoint
  static const String mobilePredictEndpoint = '/api/predict/mobile';
  static const String healthEndpoint = '/api/health';
  static const String modelsEndpoint = '/api/models';
  static const String metricsEndpoint = '/api/model-metrics';

  // Dedicated Mobile Architecture Information
  static const String assignedModelName = 'MobileNetV2';
  static const String assignedModelPlatform = 'Mobile Application';
  static const String inputResolution = '224x224 RGB';
  static const int numClasses = 5;

  // ICDR Clinical Classifications
  static const List<String> icdrClasses = [
    'No Diabetic Retinopathy (No DR)',
    'Mild Non-Proliferative Retinopathy (Mild NPDR)',
    'Moderate Non-Proliferative Retinopathy (Moderate NPDR)',
    'Severe Non-Proliferative Retinopathy (Severe NPDR)',
    'Proliferative Diabetic Retinopathy (PDR)',
  ];

  static const List<String> icdrShortNames = [
    'No DR',
    'Mild NPDR',
    'Moderate NPDR',
    'Severe NPDR',
    'Proliferative DR',
  ];

  // Request timeouts
  static const Duration connectTimeout = Duration(seconds: 15);
  static const Duration receiveTimeout = Duration(seconds: 30);
}

