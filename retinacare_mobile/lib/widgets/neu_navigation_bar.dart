import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';
import '../core/theme/app_typography.dart';

/// RetinaCare AI - Compact Floating Dark Neumorphic Bottom Navigation Bar
/// Obsidian raised outer surface with champagne gold active indicator and tactile inset response.
class NeuNavigationBar extends StatelessWidget {
  final int currentIndex;
  final ValueChanged<int> onTap;

  const NeuNavigationBar({
    super.key,
    required this.currentIndex,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.subtleBorder, width: 0.8),
        boxShadow: AppColors.neuRaised(offset: 4.0, blur: 9.0),
      ),
      child: Container(
        height: 62,
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            _buildNavItem(0, Icons.home_outlined, Icons.home_rounded, 'Home'),
            _buildNavItem(1, Icons.crop_free_rounded, Icons.center_focus_strong_rounded, 'Scan', isScan: true),
            _buildNavItem(2, Icons.history_rounded, Icons.history_rounded, 'History'),
            _buildNavItem(3, Icons.auto_awesome_outlined, Icons.auto_awesome_rounded, 'Assistant'),
            _buildNavItem(4, Icons.settings_outlined, Icons.settings_rounded, 'Settings'),
          ],
        ),
      ),
    );
  }

  Widget _buildNavItem(
    int index,
    IconData unselectedIcon,
    IconData selectedIcon,
    String label, {
    bool isScan = false,
  }) {
    final bool isSelected = currentIndex == index;

    return Expanded(
      child: InkWell(
        onTap: () => onTap(index),
        borderRadius: BorderRadius.circular(14),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          padding: const EdgeInsets.symmetric(vertical: 2, horizontal: 2),
          decoration: isSelected
              ? BoxDecoration(
                  color: AppColors.surfacePressed,
                  borderRadius: BorderRadius.circular(14),
                  gradient: AppColors.insetGradient,
                  border: Border.all(
                    color: AppColors.primaryGold.withValues(alpha: 0.4),
                    width: 0.8,
                  ),
                )
              : null,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                isSelected ? selectedIcon : unselectedIcon,
                size: isScan ? 20 : 18,
                color: isSelected ? AppColors.primaryGold : AppColors.textMuted,
              ),
              const SizedBox(height: 2),
              Text(
                label,
                style: AppTypography.caption.copyWith(
                  fontSize: 10,
                  height: 1.1,
                  fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
                  color: isSelected ? AppColors.textPrimary : AppColors.textMuted,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
