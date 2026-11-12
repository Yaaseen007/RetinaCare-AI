import 'package:flutter/material.dart';
import 'neu_progress_bar.dart';

/// RetinaCare AI - Backward compatible wrapper for NeuProgressBar
class ProbabilityBar extends StatelessWidget {
  final int stageIndex;
  final String stageName;
  final double probability;
  final bool isPredictedClass;

  const ProbabilityBar({
    super.key,
    int? classId,
    String? label,
    double? percentage,
    bool? isTopPrediction,
    int? stageIndex,
    String? stageName,
    double? probability,
    bool? isPredictedClass,
  })  : stageIndex = stageIndex ?? classId ?? 0,
        stageName = stageName ?? label ?? '',
        probability = probability ?? percentage ?? 0.0,
        isPredictedClass = isPredictedClass ?? isTopPrediction ?? false;

  @override
  Widget build(BuildContext context) {
    return NeuProgressBar(
      stageIndex: stageIndex,
      stageName: stageName,
      probability: probability,
      isPredictedClass: isPredictedClass,
    );
  }
}
