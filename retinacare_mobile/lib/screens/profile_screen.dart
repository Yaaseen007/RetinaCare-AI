import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';
import '../core/theme/app_typography.dart';
import '../widgets/neu_card.dart';
import '../widgets/neu_button.dart';
import '../widgets/neu_chip.dart';
import '../widgets/neu_input_field.dart';
import '../widgets/neu_icon_button.dart';
import '../services/storage_service.dart';
import '../services/api_service.dart';

/// RetinaCare AI - Neumorphic Settings & Profile Screen
class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final TextEditingController _urlCtrl = TextEditingController();
  final TextEditingController _clinicianCtrl = TextEditingController();
  final TextEditingController _facilityCtrl = TextEditingController();

  String _currentBaseUrl = '';
  String _clinicianName = '';
  String _clinicFacility = '';

  bool _isTesting = false;
  String? _testResultText;
  bool _testSuccess = false;

  @override
  void initState() {
    super.initState();
    _loadPreferences();
  }

  @override
  void dispose() {
    _urlCtrl.dispose();
    _clinicianCtrl.dispose();
    _facilityCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadPreferences() async {
    final url = await StorageService.getBaseUrl();
    final clinician = await StorageService.getClinicianName();
    final facility = await StorageService.getClinicFacility();

    if (mounted) {
      setState(() {
        _currentBaseUrl = url;
        _clinicianName = clinician;
        _clinicFacility = facility;
        _urlCtrl.text = url;
        _clinicianCtrl.text = clinician;
        _facilityCtrl.text = facility;
      });
    }
  }

  Future<void> _saveBaseUrl(String newUrl) async {
    final trimmed = newUrl.trim();
    if (trimmed.isEmpty) return;
    await StorageService.setBaseUrl(trimmed);
    setState(() {
      _currentBaseUrl = trimmed;
      _testResultText = null;
    });
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('API Base URL updated'),
          duration: Duration(seconds: 2),
          behavior: SnackBarBehavior.floating,
        ),
      );
    }
  }

  Future<void> _testConnection() async {
    setState(() {
      _isTesting = true;
      _testResultText = 'Connecting to backend...';
      _testSuccess = false;
    });

    try {
      final tStart = DateTime.now();
      final health = await ApiService.checkHealth();
      final pingMs = DateTime.now().difference(tStart).inMilliseconds;

      if (mounted) {
        setState(() {
          _isTesting = false;
          _testSuccess = true;
          _testResultText =
              'Connected: ${health['app_name'] ?? 'RetinaCare Backend'} (${health['version'] ?? 'v1.0.0'})\n'
              'Mobile Model: ${health['mobile_model'] ?? 'MobileNetV2'} • Latency: ${pingMs}ms';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isTesting = false;
          _testSuccess = false;
          _testResultText = 'Connection Failed: $e';
        });
      }
    }
  }

  void _showEditProfileDialog() {
    _clinicianCtrl.text = _clinicianName;
    _facilityCtrl.text = _clinicFacility;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppColors.surfaceElevated,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text('Edit Profile', style: AppTypography.sectionTitle),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: _clinicianCtrl,
              style: AppTypography.body.copyWith(color: AppColors.textPrimary),
              decoration: const InputDecoration(
                labelText: 'Clinician / User Name',
                labelStyle: TextStyle(color: AppColors.textSecondary),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _facilityCtrl,
              style: AppTypography.body.copyWith(color: AppColors.textPrimary),
              decoration: const InputDecoration(
                labelText: 'Clinic / Facility',
                labelStyle: TextStyle(color: AppColors.textSecondary),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Cancel', style: AppTypography.caption.copyWith(color: AppColors.textSecondary)),
          ),
          TextButton(
            onPressed: () async {
              await StorageService.setClinicianName(_clinicianCtrl.text);
              await StorageService.setClinicFacility(_facilityCtrl.text);
              _loadPreferences();
              if (context.mounted) Navigator.pop(context);
            },
            child: Text('Save', style: AppTypography.caption.copyWith(color: AppColors.primaryGold, fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(18, 14, 18, 90),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Settings & Profile', style: AppTypography.screenTitle),
          const SizedBox(height: 2),
          Text('Platform Configuration & System Transparency', style: AppTypography.caption),
          const SizedBox(height: 16),

          // 1. Profile Section
          _buildProfileCard(),
          const SizedBox(height: 14),

          // 2. Backend Connection Section
          _buildBackendConfigSection(),
          const SizedBox(height: 14),

          // 3. Model Architecture Section (Simple two-row layout)
          _buildArchitectureCard(),
          const SizedBox(height: 14),

          // 4. About Section
          _buildAboutCard(),
        ],
      ),
    );
  }

  Widget _buildProfileCard() {
    return NeuCard(
      padding: const EdgeInsets.all(16),
      borderRadius: 16,
      child: Row(
        children: [
          NeuCard(
            isInset: true,
            padding: const EdgeInsets.all(10),
            borderRadius: 12,
            child: const Icon(Icons.person_outline_rounded, color: AppColors.primaryGold, size: 22),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _clinicianName.isEmpty ? 'Clinician' : _clinicianName,
                  style: AppTypography.cardTitle,
                ),
                const SizedBox(height: 2),
                Text(
                  _clinicFacility.isEmpty ? 'Retinal Screening Facility' : _clinicFacility,
                  style: AppTypography.caption,
                ),
              ],
            ),
          ),
          NeuIconButton(
            icon: Icons.edit_outlined,
            iconColor: AppColors.primaryGold,
            size: 36,
            iconSize: 17,
            onTap: _showEditProfileDialog,
          ),
        ],
      ),
    );
  }

  Widget _buildBackendConfigSection() {
    return NeuCard(
      padding: const EdgeInsets.all(16),
      borderRadius: 16,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.dns_outlined, color: AppColors.primaryGold, size: 18),
              const SizedBox(width: 8),
              Text('Backend Connection', style: AppTypography.sectionTitle.copyWith(fontSize: 15)),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            'Configure host URL for API requests. Do not use localhost on physical Android devices.',
            style: AppTypography.bodySmall,
          ),
          const SizedBox(height: 10),

          // Active URL string (Inset container)
          NeuCard(
            isInset: true,
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            borderRadius: 8,
            child: Row(
              children: [
                const Icon(Icons.link_rounded, size: 14, color: AppColors.mutedGold),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    _currentBaseUrl,
                    style: AppTypography.monoValue.copyWith(fontSize: 11.5),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),

          // Inset URL Input Field + Save Button
          Row(
            children: [
              Expanded(
                child: NeuInputField(
                  controller: _urlCtrl,
                  hintText: 'http://192.168.1.6:8000',
                  height: 44,
                  textStyle: AppTypography.monoValue.copyWith(fontSize: 12, color: AppColors.textPrimary),
                ),
              ),
              const SizedBox(width: 8),
              NeuButton(
                label: 'Save',
                height: 44,
                borderRadius: 10,
                onPressed: () => _saveBaseUrl(_urlCtrl.text),
              ),
            ],
          ),
          const SizedBox(height: 14),

          // Compact Presets
          Text('Host Presets', style: AppTypography.caption.copyWith(fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: [
              NeuChip(
                label: 'LAN (192.168.1.6:8000)',
                isSelected: _currentBaseUrl == 'http://192.168.1.6:8000',
                onTap: () {
                  _urlCtrl.text = 'http://192.168.1.6:8000';
                  _saveBaseUrl('http://192.168.1.6:8000');
                },
              ),
              NeuChip(
                label: 'Emulator (10.0.2.2:8000)',
                isSelected: _currentBaseUrl == 'http://10.0.2.2:8000',
                onTap: () {
                  _urlCtrl.text = 'http://10.0.2.2:8000';
                  _saveBaseUrl('http://10.0.2.2:8000');
                },
              ),
              NeuChip(
                label: 'Localhost (127.0.0.1:8000)',
                isSelected: _currentBaseUrl == 'http://127.0.0.1:8000',
                onTap: () {
                  _urlCtrl.text = 'http://127.0.0.1:8000';
                  _saveBaseUrl('http://127.0.0.1:8000');
                },
              ),
            ],
          ),
          const SizedBox(height: 14),

          // Test Connection Button
          NeuButton(
            label: _isTesting ? 'Testing Connection...' : 'Test Connection',
            icon: Icons.network_check_rounded,
            isSecondary: true,
            isLoading: _isTesting,
            height: 42,
            onPressed: _testConnection,
          ),

          if (_testResultText != null) ...[
            const SizedBox(height: 10),
            NeuCard(
              isInset: true,
              padding: const EdgeInsets.all(10),
              borderRadius: 8,
              borderColor: _testSuccess
                  ? AppColors.drStage0.withValues(alpha: 0.4)
                  : AppColors.drStage4.withValues(alpha: 0.4),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    _testSuccess ? Icons.check_circle_outline_rounded : Icons.error_outline_rounded,
                    color: _testSuccess ? AppColors.drStage0 : AppColors.drStage4,
                    size: 16,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      _testResultText!,
                      style: AppTypography.caption.copyWith(
                        color: AppColors.textPrimary,
                        fontSize: 11,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildArchitectureCard() {
    return NeuCard(
      padding: const EdgeInsets.all(16),
      borderRadius: 16,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.memory_rounded, color: AppColors.primaryGold, size: 18),
              const SizedBox(width: 8),
              Text('Model Architecture', style: AppTypography.sectionTitle.copyWith(fontSize: 15)),
            ],
          ),
          const SizedBox(height: 12),
          _buildArchRow('Mobile App', 'MobileNetV2', AppColors.primaryGold),
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 8),
            child: Divider(height: 1, color: AppColors.subtleBorder),
          ),
          _buildArchRow('Website', 'VGG16', AppColors.softGold),
        ],
      ),
    );
  }

  Widget _buildArchRow(String label, String value, Color color) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: AppTypography.bodySmall),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.12),
            borderRadius: BorderRadius.circular(6),
            border: Border.all(color: color.withValues(alpha: 0.3), width: 0.7),
          ),
          child: Text(
            value,
            style: AppTypography.caption.copyWith(
              color: color,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildAboutCard() {
    return NeuCard(
      padding: const EdgeInsets.all(16),
      borderRadius: 16,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.info_outline_rounded, color: AppColors.textSecondary, size: 18),
              const SizedBox(width: 8),
              Text('About RetinaCare AI', style: AppTypography.sectionTitle.copyWith(fontSize: 15)),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            'RetinaCare AI is an intelligent clinical screening system for automated early detection and 5-stage grading of Diabetic Retinopathy.',
            style: AppTypography.bodySmall,
          ),
          const SizedBox(height: 8),
          Text(
            'Developed at Department of Computer Science & Engineering, GM University.',
            style: AppTypography.caption.copyWith(color: AppColors.textMuted),
          ),
        ],
      ),
    );
  }
}
