import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../core/theme/app_colors.dart';
import '../core/theme/app_typography.dart';
import '../widgets/neu_button.dart';
import '../widgets/neu_card.dart';
import '../widgets/neu_icon_button.dart';
import '../services/api_service.dart';
import 'results_screen.dart';

/// RetinaCare AI - Clean, Focused Neumorphic Retinal Scanner Screen
class ScannerScreen extends StatefulWidget {
  final VoidCallback onScanCompleted;

  const ScannerScreen({
    super.key,
    required this.onScanCompleted,
  });

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> with SingleTickerProviderStateMixin {
  final ImagePicker _picker = ImagePicker();

  Uint8List? _selectedImageBytes;
  String? _selectedImageName;
  bool _isAnalyzing = false;
  String? _errorMessage;

  late AnimationController _scannerAnimCtrl;
  late Animation<double> _scannerAnim;

  @override
  void initState() {
    super.initState();
    _scannerAnimCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1600),
    )..repeat(reverse: true);

    _scannerAnim = Tween<double>(begin: 0.05, end: 0.95).animate(
      CurvedAnimation(parent: _scannerAnimCtrl, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _scannerAnimCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickFromGallery() async {
    setState(() => _errorMessage = null);
    try {
      final XFile? file = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1920,
        maxHeight: 1920,
        imageQuality: 92,
      );
      if (file != null) {
        final bytes = await file.readAsBytes();
        _validateAndSetImage(bytes, file.name);
      }
    } catch (e) {
      setState(() => _errorMessage = 'Failed to load gallery image: $e');
    }
  }

  Future<void> _pickFromCamera() async {
    setState(() => _errorMessage = null);
    try {
      final XFile? file = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1920,
        maxHeight: 1920,
        imageQuality: 92,
      );
      if (file != null) {
        final bytes = await file.readAsBytes();
        _validateAndSetImage(bytes, file.name);
      }
    } catch (e) {
      setState(() => _errorMessage = 'Camera capture failed: $e');
    }
  }

  void _validateAndSetImage(Uint8List bytes, String name) {
    if (bytes.isEmpty) {
      setState(() => _errorMessage = 'Selected image is empty (0 bytes).');
      return;
    }
    if (bytes.length > 20 * 1024 * 1024) {
      setState(() => _errorMessage = 'Image exceeds maximum limit of 20MB.');
      return;
    }

    setState(() {
      _selectedImageBytes = bytes;
      _selectedImageName = name;
      _errorMessage = null;
    });
  }

  void _clearImage() {
    setState(() {
      _selectedImageBytes = null;
      _selectedImageName = null;
      _errorMessage = null;
    });
  }

  Future<void> _executeAnalysis() async {
    if (_selectedImageBytes == null || _selectedImageName == null || _isAnalyzing) {
      return;
    }

    setState(() {
      _isAnalyzing = true;
      _errorMessage = null;
    });

    try {
      final result = await ApiService.predictMobile(
        imageBytes: _selectedImageBytes!,
        filename: _selectedImageName!,
        confidenceThreshold: 50.0,
        includePreprocessed: true,
      );

      if (!mounted) return;

      setState(() => _isAnalyzing = false);

      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ResultsScreen(result: result),
        ),
      );

      widget.onScanCompleted();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isAnalyzing = false;
        _errorMessage = e.toString();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(18, 12, 18, 90),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Screen Title
          Text('Retinal Scanner', style: AppTypography.screenTitle),
          const SizedBox(height: 3),
          Text(
            'Upload a fundus photograph for clinical classification',
            style: AppTypography.bodySmall,
          ),
          const SizedBox(height: 16),

          // Error Banner if present
          if (_errorMessage != null) ...[
            _buildErrorBanner(),
            const SizedBox(height: 14),
          ],

          // Large Inset Image Preview
          _buildPreviewArea(),
          const SizedBox(height: 14),

          // Quality Guidance
          _buildQualityGuidance(),
          const SizedBox(height: 24),

          // Prominent Analyze Button
          NeuButton(
            label: _isAnalyzing ? 'Analyzing with MobileNetV2...' : 'Analyze with MobileNetV2',
            icon: Icons.biotech_rounded,
            isLoading: _isAnalyzing,
            onPressed: (_selectedImageBytes != null && !_isAnalyzing) ? _executeAnalysis : null,
          ),
        ],
      ),
    );
  }

  Widget _buildErrorBanner() {
    return NeuCard(
      isInset: true,
      padding: const EdgeInsets.all(12.0),
      borderRadius: 12,
      borderColor: AppColors.drStage4.withValues(alpha: 0.4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.error_outline_rounded, color: AppColors.drStage4, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              _errorMessage!,
              style: AppTypography.bodySmall.copyWith(color: AppColors.textPrimary),
            ),
          ),
          InkWell(
            onTap: () => setState(() => _errorMessage = null),
            child: const Icon(Icons.close, size: 16, color: AppColors.textMuted),
          ),
        ],
      ),
    );
  }

  // --------------------------------------------------------------------------
  // Inset Image Preview Area
  // --------------------------------------------------------------------------
  Widget _buildPreviewArea() {
    return NeuCard(
      isInset: true,
      padding: EdgeInsets.zero,
      borderRadius: 18,
      height: 290,
      width: double.infinity,
      child: Stack(
        alignment: Alignment.center,
        children: [
          if (_selectedImageBytes != null)
            ClipRRect(
              borderRadius: BorderRadius.circular(18),
              child: Image.memory(
                _selectedImageBytes!,
                fit: BoxFit.contain,
                width: double.infinity,
                height: double.infinity,
              ),
            )
          else
            Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                NeuCard(
                  padding: const EdgeInsets.all(14),
                  borderRadius: 16,
                  child: const Icon(
                    Icons.photo_library_rounded,
                    size: 32,
                    color: AppColors.primaryGold,
                  ),
                ),
                const SizedBox(height: 14),
                Text(
                  'No retinal image selected',
                  style: AppTypography.cardTitle.copyWith(fontSize: 14.5),
                ),
                const SizedBox(height: 4),
                Text(
                  'Upload a color retinal fundus photograph',
                  style: AppTypography.caption,
                ),
                const SizedBox(height: 16),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Flexible(
                        child: NeuButton(
                          label: 'Select Image',
                          icon: Icons.photo_library_outlined,
                          height: 40,
                          borderRadius: 10,
                          isSecondary: true,
                          onPressed: _pickFromGallery,
                        ),
                      ),
                      const SizedBox(width: 10),
                      NeuIconButton(
                        icon: Icons.camera_alt_outlined,
                        tooltip: 'Capture Camera',
                        size: 40,
                        iconSize: 18,
                        onTap: _pickFromCamera,
                      ),
                    ],
                  ),
                ),
              ],
            ),

          // Replace and Remove actions when image is selected
          if (_selectedImageBytes != null && !_isAnalyzing)
            Positioned(
              top: 10,
              right: 10,
              child: Row(
                children: [
                  NeuIconButton(
                    icon: Icons.refresh_rounded,
                    tooltip: 'Replace Image',
                    size: 36,
                    iconSize: 17,
                    iconColor: AppColors.primaryGold,
                    onTap: _pickFromGallery,
                  ),
                  const SizedBox(width: 8),
                  NeuIconButton(
                    icon: Icons.delete_outline_rounded,
                    tooltip: 'Remove',
                    size: 36,
                    iconSize: 17,
                    iconColor: AppColors.drStage4,
                    onTap: _clearImage,
                  ),
                ],
              ),
            ),

          // Subtle Gold Scanning Laser Line Overlay during inference
          if (_isAnalyzing)
            AnimatedBuilder(
              animation: _scannerAnimCtrl,
              builder: (context, child) {
                return Stack(
                  children: [
                    Container(
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.5),
                        borderRadius: BorderRadius.circular(18),
                      ),
                    ),
                    Align(
                      alignment: Alignment(0, (_scannerAnim.value * 2) - 1.0),
                      child: Container(
                        height: 2.5,
                        decoration: BoxDecoration(
                          boxShadow: [
                            BoxShadow(
                              color: AppColors.primaryGold.withValues(alpha: 0.8),
                              blurRadius: 10,
                              spreadRadius: 2,
                            ),
                          ],
                          gradient: const LinearGradient(
                            colors: [Colors.transparent, AppColors.primaryGold, Colors.transparent],
                          ),
                        ),
                      ),
                    ),
                    Center(
                      child: NeuCard(
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                        borderRadius: 12,
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2, color: AppColors.primaryGold),
                            ),
                            const SizedBox(width: 10),
                            Text(
                              'Analyzing with MobileNetV2...',
                              style: AppTypography.caption.copyWith(
                                color: AppColors.textPrimary,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                );
              },
            ),
        ],
      ),
    );
  }

  Widget _buildQualityGuidance() {
    return Row(
      children: [
        const Icon(Icons.info_outline_rounded, size: 14, color: AppColors.textMuted),
        const SizedBox(width: 6),
        Expanded(
          child: Text(
            'Ensure circular fundus aperture is centered with clear vascular visibility.',
            style: AppTypography.caption.copyWith(fontSize: 11),
          ),
        ),
      ],
    );
  }
}
