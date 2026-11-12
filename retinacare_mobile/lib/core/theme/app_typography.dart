import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'app_colors.dart';

/// RetinaCare AI - Luxury Neumorphic Typography System
/// Powered by Google Fonts Inter with warm ivory text and champagne gold accents.
class AppTypography {
  AppTypography._();

  // Screen Title (22-24px, warm ivory)
  static TextStyle get screenTitle => GoogleFonts.inter(
        fontSize: 23,
        fontWeight: FontWeight.w700,
        color: AppColors.textPrimary,
        letterSpacing: -0.4,
      );

  // Screen Title in Champagne Gold
  static TextStyle get screenTitleGold => GoogleFonts.inter(
        fontSize: 23,
        fontWeight: FontWeight.w700,
        color: AppColors.primaryGold,
        letterSpacing: -0.4,
      );

  // Section Title (16-18px)
  static TextStyle get sectionTitle => GoogleFonts.inter(
        fontSize: 17,
        fontWeight: FontWeight.w600,
        color: AppColors.textPrimary,
        letterSpacing: -0.2,
      );

  // Card Header / Subheading (15px)
  static TextStyle get cardTitle => GoogleFonts.inter(
        fontSize: 15,
        fontWeight: FontWeight.w600,
        color: AppColors.textPrimary,
      );

  // Body Text (14-15px)
  static TextStyle get body => GoogleFonts.inter(
        fontSize: 14.5,
        fontWeight: FontWeight.w400,
        color: AppColors.textPrimary,
        height: 1.45,
      );

  // Body Small / Secondary (12-13px)
  static TextStyle get bodySmall => GoogleFonts.inter(
        fontSize: 12.5,
        fontWeight: FontWeight.w400,
        color: AppColors.textSecondary,
        height: 1.4,
      );

  // Metadata / Caption (11-12px)
  static TextStyle get caption => GoogleFonts.inter(
        fontSize: 11.5,
        fontWeight: FontWeight.w500,
        color: AppColors.textMuted,
        letterSpacing: 0.2,
      );

  // Metric Value in Champagne Gold (20-24px)
  static TextStyle get metricValue => GoogleFonts.inter(
        fontSize: 21,
        fontWeight: FontWeight.w700,
        color: AppColors.primaryGold,
        letterSpacing: -0.5,
      );

  // Monospace Values
  static TextStyle get monoValue => GoogleFonts.jetBrainsMono(
        fontSize: 13,
        fontWeight: FontWeight.w600,
        color: AppColors.primaryGold,
      );

  // Compatibility aliases
  static TextStyle get displayLarge => screenTitleGold;
  static TextStyle get displayMedium => screenTitle;
  static TextStyle get headlineLarge => screenTitle;
  static TextStyle get headlineMedium => sectionTitle;
  static TextStyle get titleLarge => cardTitle;
  static TextStyle get bodyLarge => body;
  static TextStyle get bodyMedium => bodySmall;
  static TextStyle get labelLarge => GoogleFonts.inter(
        fontSize: 14,
        fontWeight: FontWeight.w600,
        color: AppColors.textPrimary,
      );
  static TextStyle get labelSmall => caption;
}
