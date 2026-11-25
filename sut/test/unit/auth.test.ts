import { describe, expect, it } from 'vitest';
import {
  safeRedirectPath,
  validateEmail,
  validateLogin,
  validatePassword,
  validateRegistration,
} from '../../src/modules/auth/logic';

describe('auth logic', () => {
  it('validates email format', () => {
    expect(validateEmail('  Alice@Example.com ')).toBeNull();
    expect(validateEmail('')).toBe('Email is required');
    expect(validateEmail('alice@example')).toBe('Enter a valid email address');
    expect(validateEmail('alice example.com')).toBe('Enter a valid email address');
  });

  it('enforces password length and character classes', () => {
    expect(validatePassword('Password1')).toBeNull();
    expect(validatePassword('Abcdef12')).toBeNull();
    expect(validatePassword('Abcde12')).toMatch(/at least 8/);
    expect(validatePassword('a1'.repeat(33))).toMatch(/at most 64/);
    expect(validatePassword('abcdefgh')).toMatch(/letter and one digit/);
    expect(validatePassword('12345678')).toMatch(/letter and one digit/);
  });

  it('collects registration errors per field', () => {
    expect(
      validateRegistration({ name: 'Al', email: 'al@example.com', password: 'Secret123', confirmPassword: 'Secret123' }),
    ).toEqual({});
    const errors = validateRegistration({ name: ' ', email: 'x', password: 'short', confirmPassword: 'other' });
    expect(Object.keys(errors).sort()).toEqual(['confirmPassword', 'email', 'name', 'password']);
  });

  it('requires both login fields', () => {
    expect(validateLogin({ email: '', password: '' })).toEqual({
      email: 'Email is required',
      password: 'Password is required',
    });
  });

  it('allows only local redirect targets', () => {
    expect(safeRedirectPath('/checkout')).toBe('/checkout');
    expect(safeRedirectPath('//evil.com')).toBe('/catalog');
    expect(safeRedirectPath('https://evil.com')).toBe('/catalog');
    expect(safeRedirectPath(undefined)).toBe('/catalog');
  });
});
