import 'package:flutter/material.dart';
import 'neu_card.dart';

/// RetinaCare AI - Backward compatible wrapper for NeuCard
class GlassContainer extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;
  final double borderRadius;
  final double blur;
  final bool enableBlur;
  final Color? surfaceColor;
  final Color? borderColor;
  final Gradient? gradient;
  final VoidCallback? onTap;

  const GlassContainer({
    super.key,
    required this.child,
    this.padding,
    this.margin,
    this.borderRadius = 16.0,
    this.blur = 10.0,
    this.enableBlur = false,
    this.surfaceColor,
    this.borderColor,
    this.gradient,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return NeuCard(
      padding: padding ?? const EdgeInsets.all(16.0),
      margin: margin,
      borderRadius: borderRadius,
      surfaceColor: surfaceColor,
      borderColor: borderColor,
      onTap: onTap,
      child: child,
    );
  }
}
