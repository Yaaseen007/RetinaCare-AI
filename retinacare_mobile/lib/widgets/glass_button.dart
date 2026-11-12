import 'package:flutter/material.dart';
import 'neu_button.dart';

/// RetinaCare AI - Backward compatible wrapper for NeuButton
class GlassButton extends StatelessWidget {
  final String label;
  final VoidCallback? onPressed;
  final IconData? icon;
  final bool isSecondary;
  final bool isDestructive;
  final bool isLoading;
  final double height;
  final double? width;
  final double borderRadius;

  const GlassButton({
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
  Widget build(BuildContext context) {
    return NeuButton(
      label: label,
      onPressed: onPressed,
      icon: icon,
      isSecondary: isSecondary,
      isDestructive: isDestructive,
      isLoading: isLoading,
      height: height,
      width: width,
      borderRadius: borderRadius,
    );
  }
}
