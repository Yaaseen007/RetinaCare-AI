import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import '../models/prediction_result.dart';
import '../core/constants/api_constants.dart';
import 'storage_service.dart';

/// Custom Exception types for clear diagnostic feedback
class ApiException implements Exception {
  final String message;
  final int? statusCode;
  ApiException(this.message, [this.statusCode]);
  @override
  String toString() => message;
}

class NetworkConnectionException extends ApiException {
  final String targetUrl;
  NetworkConnectionException(super.message, this.targetUrl);
}

/// RetinaCare AI - API Client Service
/// Reuses existing FastAPI backend with dedicated MobileNetV2 routing.
class ApiService {
  ApiService._();

  /// Checks server health and verifies MobileNetV2 availability
  static Future<Map<String, dynamic>> checkHealth() async {
    final baseUrl = await StorageService.getBaseUrl();
    final uri = Uri.parse('$baseUrl${ApiConstants.healthEndpoint}');

    try {
      final response = await http.get(uri).timeout(ApiConstants.connectTimeout);
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        throw ApiException(
          'Backend returned HTTP ${response.statusCode}: ${response.body}',
          response.statusCode,
        );
      }
    } on SocketException catch (e) {
      throw NetworkConnectionException(
        'Cannot connect to RetinaCare backend at $baseUrl. Ensure FastAPI is running on host 0.0.0.0:8000.\n(Error: ${e.message})',
        baseUrl,
      );
    } on TimeoutException {
      throw NetworkConnectionException(
        'Connection timed out while contacting $baseUrl. Please verify IP address and firewall settings.',
        baseUrl,
      );
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException('Unexpected network error: $e');
    }
  }

  /// Sends retinal image to dedicated mobile prediction endpoint: POST /api/predict/mobile
  /// Guarantees MobileNetV2 inference execution without fallback or fabrication.
  static Future<PredictionResult> predictMobile({
    required Uint8List imageBytes,
    required String filename,
    double confidenceThreshold = 50.0,
    bool includePreprocessed = true,
  }) async {
    final baseUrl = await StorageService.getBaseUrl();
    final uri = Uri.parse('$baseUrl${ApiConstants.mobilePredictEndpoint}');

    try {
      final request = http.MultipartRequest('POST', uri);

      // Attach file bytes
      final multipartFile = http.MultipartFile.fromBytes(
        'file',
        imageBytes,
        filename: filename,
      );
      request.files.add(multipartFile);

      // Form fields
      request.fields['confidence_threshold'] = confidenceThreshold.toString();
      request.fields['include_preprocessed_image'] = includePreprocessed.toString();

      final streamedResponse = await request.send().timeout(ApiConstants.receiveTimeout);
      final responseBody = await streamedResponse.stream.bytesToString();

      if (streamedResponse.statusCode == 200) {
        final Map<String, dynamic> data = jsonDecode(responseBody) as Map<String, dynamic>;

        // Encode local image as base64 to store in local result preview
        final localB64 = 'data:image/png;base64,${base64Encode(imageBytes)}';
        final result = PredictionResult.fromJson(data, localImageBase64: localB64);

        // Save scan automatically to local History persistence
        await StorageService.saveScan(result);

        return result;
      } else {
        String detailMessage = responseBody;
        try {
          final errorJson = jsonDecode(responseBody);
          if (errorJson is Map && errorJson.containsKey('detail')) {
            detailMessage = errorJson['detail'].toString();
          }
        } catch (_) {}

        throw ApiException(
          detailMessage,
          streamedResponse.statusCode,
        );
      }
    } on SocketException catch (e) {
      throw NetworkConnectionException(
        'Failed to connect to backend at $baseUrl.\nVerify your network connection or update the Base URL in Profile Settings.\n($e)',
        baseUrl,
      );
    } on TimeoutException {
      throw NetworkConnectionException(
        'Inference request timed out. The server took longer than 30 seconds to respond.',
        baseUrl,
      );
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException('Inference communication failure: $e');
    }
  }
}

