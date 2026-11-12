import 'dart:convert';
import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';
import '../core/theme/app_typography.dart';
import '../widgets/neu_card.dart';
import '../widgets/neu_button.dart';
import '../widgets/neu_progress_bar.dart';
import '../models/prediction_result.dart';
import 'assistant_screen.dart';

/// RetinaCare AI - Elegant Neumorphic Clinical Results Screen
class ResultsScreen extends StatelessWidget {
  final PredictionResult result;

  const ResultsScreen({
    super.key,
    required this.result,
  });

  @override
  Widget build(BuildContext context) {
    final Color stageColor = AppColors.getColorForStage(result.classId);

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: Text('Diagnostic Result', style: AppTypography.screenTitle.copyWith(fontSize: 18)),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded, size: 18),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(18, 8, 18, 30),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 1 & 2. Predicted ICDR Stage & Actual Confidence Header Card
            _buildClassificationHeader(stageColor),
            const SizedBox(height: 14),

            // Image Preview (Original and Enhanced)
            _buildImagePreviewSection(),
            const SizedBox(height: 14),

            // 3. Five-Stage Probability Visualization
            _buildProbabilitySection(),
            const SizedBox(height: 14),

            // 4 & 5. Educational Explanation & Recommended Next Steps
            _buildClinicalGuidance(stageColor),
            const SizedBox(height: 14),

            // 6. Medical Disclaimer
            _buildDisclaimer(),
            const SizedBox(height: 20),

            // Actions
            NeuButton(
              label: 'Consult Assistant on Stage ${result.classId}',
              icon: Icons.auto_awesome_rounded,
              isSecondary: true,
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => AssistantScreen(
                      initialQuery: 'Explain clinical findings for ${result.shortName}',
                    ),
                  ),
                );
              },
            ),
            const SizedBox(height: 10),
            NeuButton(
              label: 'Done',
              onPressed: () => Navigator.pop(context),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildClassificationHeader(Color stageColor) {
    return NeuCard(
      padding: const EdgeInsets.all(18.0),
      borderRadius: 16.0,
      isGoldAccent: true,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                decoration: BoxDecoration(
                  color: stageColor.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: stageColor.withValues(alpha: 0.4), width: 0.8),
                ),
                child: Text(
                  'STAGE ${result.classId} OF 4',
                  style: AppTypography.caption.copyWith(
                    color: stageColor,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  '${result.latencyMs.toStringAsFixed(0)} ms · ${result.modelName}',
                  style: AppTypography.caption.copyWith(fontSize: 11),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            result.className,
            style: AppTypography.sectionTitle.copyWith(fontSize: 18, height: 1.3),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Text(
                '${result.confidencePct.toStringAsFixed(1)}%',
                style: AppTypography.metricValue.copyWith(
                  color: stageColor,
                  fontSize: 28,
                ),
              ),
              const SizedBox(width: 10),
              Flexible(
                child: Text(
                  result.isLowConfidence ? 'Borderline Confidence (<50%)' : 'Confidence Score',
                  style: AppTypography.caption.copyWith(
                    color: result.isLowConfidence ? AppColors.drStage2 : AppColors.textSecondary,
                    fontSize: 11.5,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildImagePreviewSection() {
    final bool hasPreprocessed = result.preprocessedImageBase64 != null &&
        result.preprocessedImageBase64!.startsWith('data:image');
    final bool hasLocalOriginal = result.localImageBase64 != null &&
        result.localImageBase64!.startsWith('data:image');

    if (!hasPreprocessed && !hasLocalOriginal) {
      return const SizedBox.shrink();
    }

    return NeuCard(
      padding: const EdgeInsets.all(14.0),
      borderRadius: 16.0,
      child: Row(
        children: [
          if (hasLocalOriginal)
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Captured Fundus', style: AppTypography.caption),
                  const SizedBox(height: 6),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(10),
                    child: Container(
                      height: 120,
                      width: double.infinity,
                      decoration: BoxDecoration(
                        color: AppColors.surfacePressed,
                        border: Border.all(color: AppColors.subtleBorder),
                      ),
                      child: Image.memory(
                        base64Decode(result.localImageBase64!.split(',').last),
                        fit: BoxFit.contain,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          if (hasLocalOriginal && hasPreprocessed) const SizedBox(width: 12),
          if (hasPreprocessed)
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Enhanced Retina', style: AppTypography.caption),
                  const SizedBox(height: 6),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(10),
                    child: Container(
                      height: 120,
                      width: double.infinity,
                      decoration: BoxDecoration(
                        color: AppColors.surfacePressed,
                        border: Border.all(color: AppColors.subtleBorder),
                      ),
                      child: Image.memory(
                        base64Decode(result.preprocessedImageBase64!.split(',').last),
                        fit: BoxFit.contain,
                      ),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildProbabilitySection() {
    return NeuCard(
      padding: const EdgeInsets.all(16.0),
      borderRadius: 16.0,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Five-Stage Class Probabilities', style: AppTypography.cardTitle),
          const SizedBox(height: 10),
          ...List.generate(5, (index) {
            final double pct = result.rawProbabilities.length > index
                ? result.rawProbabilities[index] * 100.0
                : 0.0;
            return NeuProgressBar(
              stageIndex: index,
              stageName: _getShortNameForClass(index),
              probability: pct,
              isPredictedClass: result.classId == index,
            );
          }),
        ],
      ),
    );
  }

  String _getShortNameForClass(int classId) {
    const list = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR'];
    return (classId >= 0 && classId < list.length) ? list[classId] : 'Stage $classId';
  }

  Widget _buildClinicalGuidance(Color stageColor) {
    return NeuCard(
      padding: const EdgeInsets.all(16.0),
      borderRadius: 16.0,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Clinical Guidance', style: AppTypography.cardTitle),
          const SizedBox(height: 10),
          Text(
            'Pathology Findings:',
            style: AppTypography.caption.copyWith(fontWeight: FontWeight.w600, color: AppColors.textSecondary),
          ),
          const SizedBox(height: 3),
          Text(result.description, style: AppTypography.bodySmall),
          const SizedBox(height: 12),
          Text(
            'Recommended Next Steps:',
            style: AppTypography.caption.copyWith(fontWeight: FontWeight.w600, color: stageColor),
          ),
          const SizedBox(height: 3),
          Text(result.recommendation, style: AppTypography.bodySmall),
        ],
      ),
    );
  }

  Widget _buildDisclaimer() {
    return Text(
      'Medical Notice: RetinaCare AI outputs are clinical decision-support predictions and must be confirmed by an eye care specialist.',
      style: AppTypography.caption.copyWith(fontSize: 10.5, height: 1.35),
      textAlign: TextAlign.center,
    );
  }
}
