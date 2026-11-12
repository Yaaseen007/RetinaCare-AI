import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';
import '../core/theme/app_typography.dart';

/// RetinaCare AI - Neumorphic 5-Class Probability Bar
/// Features an inset track with smooth clinical-stage progress fill.
class NeuProgressBar extends StatelessWidget {
  final int stageIndex;
  final String stageName;
  final double probability; // 0.0 to 1.0 or 0.0 to 100.0
  final bool isPredictedClass;

  const NeuProgressBar({
    super.key,
    required this.stageIndex,
    required this.stageName,
    required this.probability,
    this.isPredictedClass = false,
  });

  @override
  Widget build(BuildContext context) {
    final double normalizedProb = probability > 1.0 ? probability / 100.0 : probability;
    final double clampedProb = normalizedProb.clamp(0.0, 1.0);
    final double pct = clampedProb * 100.0;
    final Color stageColor = AppColors.getColorForStage(stageIndex);

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4.5),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Row(
                  children: [
                    Container(
                      width: 7,
                      height: 7,
                      decoration: BoxDecoration(
                        color: stageColor,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Flexible(
                      child: Text(
                        stageName,
                        style: AppTypography.caption.copyWith(
                          color: isPredictedClass ? AppColors.textPrimary : AppColors.textSecondary,
                          fontWeight: isPredictedClass ? FontWeight.w700 : FontWeight.w400,
                          fontSize: 12,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    if (isPredictedClass) ...[
                      const SizedBox(width: 6),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1.5),
                        decoration: BoxDecoration(
                          color: AppColors.primaryGold.withValues(alpha: 0.15),
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(color: AppColors.primaryGold.withValues(alpha: 0.4), width: 0.6),
                        ),
                        child: Text(
                          'PREDICTED',
                          style: AppTypography.caption.copyWith(
                            fontSize: 8.5,
                            color: AppColors.primaryGold,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 0.5,
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Text(
                '${pct.toStringAsFixed(1)}%',
                style: AppTypography.monoValue.copyWith(
                  fontSize: 12,
                  color: isPredictedClass ? stageColor : AppColors.textSecondary,
                  fontWeight: isPredictedClass ? FontWeight.w700 : FontWeight.w500,
                ),
              ),
            ],
          ),
          const SizedBox(height: 5),
          // Inset Track
          Container(
            height: 7,
            decoration: BoxDecoration(
              color: AppColors.surfacePressed,
              borderRadius: BorderRadius.circular(4),
              gradient: AppColors.insetGradient,
              border: Border.all(color: AppColors.subtleBorder.withValues(alpha: 0.6), width: 0.6),
            ),
            child: LayoutBuilder(
              builder: (context, constraints) {
                return Stack(
                  children: [
                    TweenAnimationBuilder<double>(
                      tween: Tween<double>(begin: 0.0, end: clampedProb),
                      duration: const Duration(milliseconds: 600),
                      curve: Curves.easeOutCubic,
                      builder: (context, value, _) {
                        return Container(
                          width: constraints.maxWidth * value,
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(4),
                            color: stageColor,
                            boxShadow: [
                              BoxShadow(
                                color: stageColor.withValues(alpha: 0.4),
                                blurRadius: 4,
                                offset: const Offset(0, 1),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                  ],
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
