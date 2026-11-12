import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';
import '../core/theme/app_typography.dart';

/// RetinaCare AI - Tactile Dark Neumorphic Button
/// Features raised resting state, inset pressed transition, and champagne gold accents.
class NeuButton extends StatefulWidget {
  final String label;
  final VoidCallback? onPressed;
  final IconData? icon;
  final bool isSecondary;
  final bool isDestructive;
  final bool isLoading;
  final double height;
  final double? width;
  final double borderRadius;

  const NeuButton({
    super.key,
    required this.label,
    this.onPressed,
    this.icon,
    this.isSecondary = false,
    this.isDestructive = false,
    this.isLoading = false,
    this.height = 48.0,
    this.width,
    this.borderRadius = 14.0,
  });

  @override
  State<NeuButton> createState() => _NeuButtonState();
}

class _NeuButtonState extends State<NeuButton> {
  bool _isPressed = false;

  bool get _isEnabled => widget.onPressed != null && !widget.isLoading;

  @override
  Widget build(BuildContext context) {
    final bool showGoldPrimary = !widget.isSecondary && !widget.isDestructive;

    BoxDecoration decoration;
    if (!_isEnabled) {
      decoration = BoxDecoration(
        color: AppColors.surfacePressed,
        borderRadius: BorderRadius.circular(widget.borderRadius),
        border: Border.all(color: AppColors.subtleBorder.withValues(alpha: 0.5)),
      );
    } else if (_isPressed) {
      // Inset state
      decoration = BoxDecoration(
        color: showGoldPrimary ? AppColors.mutedGold.withValues(alpha: 0.3) : AppColors.surfacePressed,
        borderRadius: BorderRadius.circular(widget.borderRadius),
        gradient: showGoldPrimary ? null : AppColors.insetGradient,
        border: Border.all(
          color: showGoldPrimary ? AppColors.primaryGold : AppColors.subtleBorder,
          width: 1.0,
        ),
      );
    } else {
      // Raised state
      if (showGoldPrimary) {
        decoration = BoxDecoration(
          borderRadius: BorderRadius.circular(widget.borderRadius),
          gradient: AppColors.goldButtonGradient,
          boxShadow: AppColors.neuRaised(offset: 3.5, blur: 8.0),
        );
      } else if (widget.isDestructive) {
        decoration = BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(widget.borderRadius),
          border: Border.all(color: AppColors.drStage4.withValues(alpha: 0.5)),
          boxShadow: AppColors.neuRaisedSoft(offset: 2.5, blur: 5.0),
        );
      } else {
        decoration = BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(widget.borderRadius),
          gradient: AppColors.raisedGradient,
          border: Border.all(color: AppColors.subtleBorder, width: 0.8),
          boxShadow: AppColors.neuRaised(offset: 3.0, blur: 6.0),
        );
      }
    }

    final Color contentColor = !_isEnabled
        ? AppColors.textDisabled
        : showGoldPrimary
            ? AppColors.textInverse
            : widget.isDestructive
                ? AppColors.drStage4
                : AppColors.primaryGold;

    return GestureDetector(
      onTapDown: _isEnabled ? (_) => setState(() => _isPressed = true) : null,
      onTapUp: _isEnabled
          ? (_) {
              setState(() => _isPressed = false);
              widget.onPressed?.call();
            }
          : null,
      onTapCancel: _isEnabled ? () => setState(() => _isPressed = false) : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 100),
        height: widget.height,
        width: widget.width,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        decoration: decoration,
        child: Center(
          child: widget.isLoading
              ? SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                    strokeWidth: 2.2,
                    valueColor: AlwaysStoppedAnimation<Color>(contentColor),
                  ),
                )
              : Row(
                  mainAxisSize: MainAxisSize.min,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    if (widget.icon != null) ...[
                      Icon(widget.icon, color: contentColor, size: 18),
                      const SizedBox(width: 8),
                    ],
                    Flexible(
                      child: Text(
                        widget.label,
                        style: AppTypography.cardTitle.copyWith(
                          color: contentColor,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.1,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
        ),
      ),
    );
  }
}

