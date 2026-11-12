import 'dart:convert';

/// RetinaCare AI - Prediction Result Model
/// Grounded strictly in the FastAPI PredictionResponse schema.
class PredictionResult {
  final String id;
  final String modelName;
  final String modelId;
  final String architectureType;
  final String backbone;
  final String trainingType;
  final String datasetName;
  final int classId;
  final String className;
  final String shortName;
  final double confidencePct;
  final double latencyMs;
  final bool isLowConfidence;
  final String confidenceStatus;
  final double confidenceThreshold;
  final Map<String, double> probabilities;
  final List<double> rawProbabilities;
  final String description;
  final String recommendation;
  final String severityColor;
  final String disclaimer;
  final String? retinalWarning;
  final String timestamp;
  final String? preprocessedImageBase64;
  final String? localImageBase64;

  PredictionResult({
    required this.id,
    required this.modelName,
    required this.modelId,
    required this.architectureType,
    required this.backbone,
    required this.trainingType,
    required this.datasetName,
    required this.classId,
    required this.className,
    required this.shortName,
    required this.confidencePct,
    required this.latencyMs,
    required this.isLowConfidence,
    required this.confidenceStatus,
    required this.confidenceThreshold,
    required this.probabilities,
    required this.rawProbabilities,
    required this.description,
    required this.recommendation,
    required this.severityColor,
    required this.disclaimer,
    this.retinalWarning,
    required this.timestamp,
    this.preprocessedImageBase64,
    this.localImageBase64,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json, {String? localImageBase64}) {
    // Parse probabilities map safely
    final rawProbMap = json['probabilities'];
    final Map<String, double> parsedProbMap = {};
    if (rawProbMap is Map) {
      rawProbMap.forEach((k, v) {
        parsedProbMap[k.toString()] = (v as num).toDouble();
      });
    }

    // Parse raw probabilities list safely
    final rawProbList = json['raw_probabilities'];
    final List<double> parsedRawList = [];
    if (rawProbList is List) {
      for (final item in rawProbList) {
        if (item is num) parsedRawList.add(item.toDouble());
      }
    }

    return PredictionResult(
      id: json['id']?.toString() ?? DateTime.now().millisecondsSinceEpoch.toString(),
      modelName: json['model_name']?.toString() ?? 'MobileNetV2',
      modelId: json['model_id']?.toString() ?? 'mobilenetv2',
      architectureType: json['architecture_type']?.toString() ?? 'Lightweight Inverted Residual CNN',
      backbone: json['backbone']?.toString() ?? 'MobileNetV2',
      trainingType: json['training_type']?.toString() ?? 'Transfer Learning',
      datasetName: json['dataset_name']?.toString() ?? 'Diabetic Retinopathy Dataset',
      classId: (json['class_id'] as num?)?.toInt() ?? 0,
      className: json['class_name']?.toString() ?? 'Unknown',
      shortName: json['short_name']?.toString() ?? 'Unknown',
      confidencePct: (json['confidence_pct'] as num?)?.toDouble() ?? 0.0,
      latencyMs: (json['latency_ms'] as num?)?.toDouble() ?? 0.0,
      isLowConfidence: json['is_low_confidence'] as bool? ?? false,
      confidenceStatus: json['confidence_status']?.toString() ?? '',
      confidenceThreshold: (json['confidence_threshold'] as num?)?.toDouble() ?? 50.0,
      probabilities: parsedProbMap,
      rawProbabilities: parsedRawList,
      description: json['description']?.toString() ?? '',
      recommendation: json['recommendation']?.toString() ?? '',
      severityColor: json['severity_color']?.toString() ?? '#10B981',
      disclaimer: json['disclaimer']?.toString() ?? '',
      retinalWarning: json['retinal_warning']?.toString(),
      timestamp: json['timestamp']?.toString() ?? DateTime.now().toIso8601String(),
      preprocessedImageBase64: json['preprocessed_image_base64']?.toString(),
      localImageBase64: localImageBase64 ?? json['local_image_base64']?.toString(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'model_name': modelName,
      'model_id': modelId,
      'architecture_type': architectureType,
      'backbone': backbone,
      'training_type': trainingType,
      'dataset_name': datasetName,
      'class_id': classId,
      'class_name': className,
      'short_name': shortName,
      'confidence_pct': confidencePct,
      'latency_ms': latencyMs,
      'is_low_confidence': isLowConfidence,
      'confidence_status': confidenceStatus,
      'confidence_threshold': confidenceThreshold,
      'probabilities': probabilities,
      'raw_probabilities': rawProbabilities,
      'description': description,
      'recommendation': recommendation,
      'severity_color': severityColor,
      'disclaimer': disclaimer,
      'retinal_warning': retinalWarning,
      'timestamp': timestamp,
      'preprocessed_image_base64': preprocessedImageBase64,
      'local_image_base64': localImageBase64,
    };
  }

  String toJsonString() => jsonEncode(toJson());

  factory PredictionResult.fromJsonString(String jsonString) {
    return PredictionResult.fromJson(jsonDecode(jsonString) as Map<String, dynamic>);
  }
}

