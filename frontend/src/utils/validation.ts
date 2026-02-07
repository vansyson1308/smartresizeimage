/**
 * Input validation utilities for the frontend.
 */

const MAX_DIMENSION = 4096;
const MIN_DIMENSION = 10;

/**
 * Validate image dimensions.
 * @returns Error message string if invalid, null if valid.
 */
export const validateDimensions = (width: number, height: number): string | null => {
  if (!Number.isFinite(width) || !Number.isFinite(height)) {
    return "Dimensions must be valid numbers";
  }
  if (width <= 0 || height <= 0) {
    return "Dimensions must be positive";
  }
  if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
    return `Maximum dimension is ${MAX_DIMENSION}px`;
  }
  if (width < MIN_DIMENSION || height < MIN_DIMENSION) {
    return `Minimum dimension is ${MIN_DIMENSION}px`;
  }
  return null; // valid
};
