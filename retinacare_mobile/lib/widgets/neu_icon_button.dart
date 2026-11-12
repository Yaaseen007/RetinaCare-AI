import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';

/// RetinaCare AI - Neumorphic Circular or Rounded Icon Button
class NeuIconButton extends StatefulWidget {
  final IconData icon;
  final VoidCallback? onTap;
  final double size;
  final double iconSize;
  final Color? iconColor;
  final bool isSelected;
  final String? tooltip;
  final double borderRadius;

  const NeuIconButton({
    super.key,
    required this.icon,
    this.onTap,
    this.size = 42.0,
    this.iconSize = 20.0,
    this.iconColor,
    this.isSelected = false,
    this.tooltip,
    this.borderRadius = 12.0,
  });

  @override
  State<NeuIconButton> createState() => _NeuIconButtonState();
}

class _NeuIconButtonState extends State<NeuIconButton> {
  bool _isPressed = false;

  @override
  Widget build(BuildContext context) {
    final effectiveColor = widget.iconColor ?? (widget.isSelected ? AppColors.primaryGold : AppColors.textSecondary);
    final insetState = _isPressed || widget.isSelected;

    Widget button = GestureDetector(
      onTapDown: widget.onTap != null ? (_) => setState(() => _isPressed = true) : null,
      onTapUp: widget.onTap != null
          ? (_) {
              setState(() => _isPressed = false);
              widget.onTap?.call();
            }
          : null,
      onTapCancel: widget.onTap != null ? () => setState(() => _isPressed = false) : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 100),
        width: widget.size,
        height: widget.size,
        decoration: insetState
            ? BoxDecoration(
                color: AppColors.surfacePressed,
                borderRadius: BorderRadius.circular(widget.borderRadius),
                gradient: AppColors.insetGradient,
                border: Border.all(
                  color: widget.isSelected ? AppColors.primaryGold : AppColors.subtleBorder,
                  width: 0.8,
                ),
              )
            : BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(widget.borderRadius),
                gradient: AppColors.raisedGradient,
                border: Border.all(color: AppColors.subtleBorder, width: 0.8),
                boxShadow: AppColors.neuRaisedSoft(offset: 2.5, blur: 5.0),
              ),
        child: Center(
          child: Icon(
            widget.icon,
            size: widget.iconSize,
            color: effectiveColor,
          ),
        ),
      ),
    );

    if (widget.tooltip != null) {
      return Tooltip(message: widget.tooltip!, child: button);
    }
    return button;
  }
}

