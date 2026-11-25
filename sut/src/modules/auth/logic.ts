// Pure validation rules for registration and login (no DB or HTTP access).

export const NAME_MIN_LENGTH = 2;
export const NAME_MAX_LENGTH = 50;
export const PASSWORD_MIN_LENGTH = 8;
export const PASSWORD_MAX_LENGTH = 64;
export const EMAIL_MAX_LENGTH = 254;

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

export type FieldErrors<K extends string> = Partial<Record<K, string>>;

export interface RegistrationInput {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
}

export interface LoginInput {
  email: string;
  password: string;
}

export function normalizeEmail(email: string): string {
  return email.trim().toLowerCase();
}

export function validateName(name: string): string | null {
  const value = name.trim();
  if (value === '') return 'Name is required';
  if (value.length < NAME_MIN_LENGTH) return `Name must be at least ${NAME_MIN_LENGTH} characters`;
  if (value.length > NAME_MAX_LENGTH) return `Name must be at most ${NAME_MAX_LENGTH} characters`;
  return null;
}

export function validateEmail(email: string): string | null {
  const value = normalizeEmail(email);
  if (value === '') return 'Email is required';
  if (value.length > EMAIL_MAX_LENGTH || !EMAIL_PATTERN.test(value)) return 'Enter a valid email address';
  return null;
}

export function validatePassword(password: string): string | null {
  if (password === '') return 'Password is required';
  if (password.length < PASSWORD_MIN_LENGTH) return `Password must be at least ${PASSWORD_MIN_LENGTH} characters`;
  if (password.length > PASSWORD_MAX_LENGTH) return `Password must be at most ${PASSWORD_MAX_LENGTH} characters`;
  if (!/[A-Za-z]/.test(password) || !/\d/.test(password)) {
    return 'Password must contain at least one letter and one digit';
  }
  return null;
}

export function validateRegistration(input: RegistrationInput): FieldErrors<keyof RegistrationInput> {
  const errors: FieldErrors<keyof RegistrationInput> = {};
  const name = validateName(input.name);
  if (name) errors.name = name;
  const email = validateEmail(input.email);
  if (email) errors.email = email;
  const password = validatePassword(input.password);
  if (password) errors.password = password;
  if (input.confirmPassword !== input.password) errors.confirmPassword = 'Passwords do not match';
  return errors;
}

export function validateLogin(input: LoginInput): FieldErrors<keyof LoginInput> {
  const errors: FieldErrors<keyof LoginInput> = {};
  if (normalizeEmail(input.email) === '') errors.email = 'Email is required';
  if (input.password === '') errors.password = 'Password is required';
  return errors;
}

export function hasErrors(errors: object): boolean {
  return Object.keys(errors).length > 0;
}

/** Allows only local paths as a post-login redirect target to prevent open redirects. */
export function safeRedirectPath(next: unknown, fallback = '/catalog'): string {
  if (typeof next !== 'string' || !next.startsWith('/')) return fallback;
  if (next.startsWith('//') || next.startsWith('/\\')) return fallback;
  return next;
}
