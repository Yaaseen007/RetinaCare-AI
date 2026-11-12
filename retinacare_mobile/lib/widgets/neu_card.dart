import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';

/// RetinaCare AI - Reusable Dark Neumorphic Card
/// Supports raised surface, inset/recessed surface, and gold accent styling.
class NeuCard extends StatelessWidget {
  final Widget? child;
  final EdgeInsetsGeometry padding;
  final EdgeInsetsGeometry? margin;
  final double borderRadius;
  final Color? surfaceColor;
  final Color? borderColor;
  final bool isInset;
  final bool isGoldAccent;
  final double? width;
  final double? height;
  final VoidCallback? onTap;

  const NeuCard({
    super.key,
    this.child,
    this.padding = const EdgeInsets.all(16.0),
    this.margin,
    this.borderRadius = 16.0,
    this.surfaceColor,
    this.borderColor,
    this.isInset = false,
    this.isGoldAccent = false,
    this.width,
    this.height,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final baseSurface = surfaceColor ?? (isInset ? AppColors.surfacePressed : AppColors.surface);
    final effectiveBorderColor = borderColor ??
        (isGoldAccent ? AppColors.primaryGold.withValues(alpha: 0.35) : AppColors.subtleBorder);

    final decoration = isInset
        ? BoxDecoration(
            color: baseSurface,
            borderRadius: BorderRadius.circular(borderRadius),
            gradient: AppColors.insetGradient,
            border: Border.all(
              color: effectiveBorderColor,
              width: isGoldAccent ? 1.0 : 0.75,
            ),
          )
        : BoxDecoration(
            color: baseSurface,
            borderRadius: BorderRadius.circular(borderRadius),
            gradient: isGoldAccent ? null : AppColors.raisedGradient,
            border: Border.all(
              color: effectiveBorderColor,
              width: isGoldAccent ? 1.0 : 0.75,
            ),
            boxShadow: AppColors.neuRaised(offset: 3.5, blur: 7.0),
          );

    Widget content = Container(
      width: width,
      height: height,
      margin: margin,
      padding: padding,
      decoration: decoration,
      child: child,
    );

    if (onTap != null) {
      return Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(borderRadius),
          splashColor: AppColors.primaryGold.withValues(alpha: 0.12),
          highlightColor: AppColors.primaryGold.withValues(alpha: 0.06),
          child: content,
        ),
      );
    }

    return content;
  }
}

