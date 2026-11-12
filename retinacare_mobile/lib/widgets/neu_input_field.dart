import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';
import '../core/theme/app_typography.dart';

/// RetinaCare AI - Inset Neumorphic Input Field
class NeuInputField extends StatefulWidget {
  final TextEditingController controller;
  final String hintText;
  final Widget? prefixIcon;
  final Widget? suffixIcon;
  final ValueChanged<String>? onSubmitted;
  final TextInputType keyboardType;
  final bool autofocus;
  final double height;
  final double borderRadius;
  final TextStyle? textStyle;

  const NeuInputField({
    super.key,
    required this.controller,
    required this.hintText,
    this.prefixIcon,
    this.suffixIcon,
    this.onSubmitted,
    this.keyboardType = TextInputType.text,
    this.autofocus = false,
    this.height = 46.0,
    this.borderRadius = 12.0,
    this.textStyle,
  });

  @override
  State<NeuInputField> createState() => _NeuInputFieldState();
}

class _NeuInputFieldState extends State<NeuInputField> {
  final FocusNode _focusNode = FocusNode();
  bool _isFocused = false;

  @override
  void initState() {
    super.initState();
    _focusNode.addListener(() {
      setState(() => _isFocused = _focusNode.hasFocus);
    });
  }

  @override
  void dispose() {
    _focusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: widget.height,
      decoration: BoxDecoration(
        color: AppColors.surfacePressed,
        borderRadius: BorderRadius.circular(widget.borderRadius),
        gradient: AppColors.insetGradient,
        border: Border.all(
          color: _isFocused ? AppColors.primaryGold : AppColors.subtleBorder,
          width: _isFocused ? 1.0 : 0.75,
        ),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 12),
      child: Row(
        children: [
          if (widget.prefixIcon != null) ...[
            widget.prefixIcon!,
            const SizedBox(width: 8),
          ],
          Expanded(
            child: TextField(
              controller: widget.controller,
              focusNode: _focusNode,
              autofocus: widget.autofocus,
              keyboardType: widget.keyboardType,
              style: widget.textStyle ??
                  AppTypography.body.copyWith(
                    color: AppColors.textPrimary,
                    fontSize: 13.5,
                  ),
              decoration: InputDecoration(
                hintText: widget.hintText,
                hintStyle: AppTypography.caption.copyWith(
                  color: AppColors.textMuted,
                  fontSize: 12.5,
                ),
                border: InputBorder.none,
                isDense: true,
                contentPadding: const EdgeInsets.symmetric(vertical: 12),
              ),
              onSubmitted: widget.onSubmitted,
            ),
          ),
          if (widget.suffixIcon != null) widget.suffixIcon!,
        ],
      ),
    );
  }
}

