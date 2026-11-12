import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';
import '../core/theme/app_typography.dart';

/// RetinaCare AI - Neumorphic Selectable Chip
/// Raised when idle, inset with champagne gold highlight when selected.
class NeuChip extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback onTap;
  final IconData? icon;
  final Color? activeColor;

  const NeuChip({
    super.key,
    required this.label,
    required this.isSelected,
    required this.onTap,
    this.icon,
    this.activeColor,
  });

  @override
  Widget build(BuildContext context) {
    final effectiveActiveColor = activeColor ?? AppColors.primaryGold;

    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(10),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 120),
        padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 6),
        decoration: isSelected
            ? BoxDecoration(
                color: AppColors.surfacePressed,
                borderRadius: BorderRadius.circular(10),
                gradient: AppColors.insetGradient,
                border: Border.all(
                  color: effectiveActiveColor,
                  width: 0.8,
                ),
              )
            : BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(10),
                gradient: AppColors.raisedGradient,
                border: Border.all(color: AppColors.subtleBorder, width: 0.75),
                boxShadow: AppColors.neuRaisedSoft(offset: 2.0, blur: 4.0),
              ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (icon != null) ...[
              Icon(
                icon,
                size: 13,
                color: isSelected ? effectiveActiveColor : AppColors.textSecondary,
              ),
              const SizedBox(width: 5),
            ],
            Flexible(
              child: Text(
                label,
                style: AppTypography.caption.copyWith(
                  color: isSelected ? effectiveActiveColor : AppColors.textSecondary,
                  fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
                  fontSize: 11,
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

