/////////////////////////////////////////////////////////////////////////////////
// This file is a C++ wrapper around wyhash: 
// https://github.com/wangyi-fudan/wyhash
// 
// Copyright (c) 2022 by Alain Espinosa.
/////////////////////////////////////////////////////////////////////////////////
// wyhash and wyrand are the ideal 64-bit hash function and PRNG respectively:
//
// solid: wyhash passed SMHasher, wyrand passed BigCrush, practrand.
// portable: 64-bit / 32-bit system, big / little endian.
// fastest: Efficient on 64-bit machines, especially for short keys.
// simplest: In the sense of code size.
// salted: We use dynamic secret to avoid intended attack.

#ifndef WYHASH_WYHASH_HPP
#define WYHASH_WYHASH_HPP

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <string>

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus > 201703L)
#include <version>
#else
#include <ciso646>
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// wyhash.h
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef WYHASH_CONDOM
	// protections that produce different results:
	// 1: normal valid behavior
	// 2: extra protection against entropy loss (probability=2^-63), aka. "blind multiplication"
	#define WYHASH_CONDOM 1
#endif

#ifndef WYHASH_32BIT_MUM
	// 0: normal version, slow on 32 bit systems
	// 1: faster on 32 bit systems but produces different results, incompatible with wy2u0k function
	#define WYHASH_32BIT_MUM 0  
#endif

// includes
#if defined(_MSC_VER) && defined(_M_X64)
	#include <intrin.h>
	#pragma intrinsic(_umul128)
#endif

// likely and unlikely macros
#if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
	#define _likely_(x)  __builtin_expect(x,1)
	#define _unlikely_(x)  __builtin_expect(x,0)
#else
	#define _likely_(x) (x)
	#define _unlikely_(x) (x)
#endif

// endian macros
#ifndef WYHASH_LITTLE_ENDIAN
	#if defined(_WIN32) || defined(__LITTLE_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
		#define WYHASH_LITTLE_ENDIAN 1
	#elif defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
		#define WYHASH_LITTLE_ENDIAN 0
	#else
		#warning could not determine endianness!Falling back to little endian.
		#define WYHASH_LITTLE_ENDIAN 1
	#endif
#endif

// Define 'byteswap64(uint64_t)' needed for 'rand::generate_stream(size_t)'
// and 'byteswap32(uint32_t)' for other functions
#if !WYHASH_LITTLE_ENDIAN
	#ifdef __cpp_lib_byteswap
		#include <bit>
		#define byteswap64(v) std::byteswap(v)
		#define byteswap32(v) std::byteswap(v)
	#elif defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
		#define byteswap64(v) __builtin_bswap64(v)
		#define byteswap32(v) __builtin_bswap32(v)
	#elif defined(_MSC_VER)
		#define byteswap64(v) _byteswap_uint64(v)
		#define byteswap32(v) _byteswap_ulong(v)
	#else
		static inline uint64_t byteswap64(uint64_t v) noexcept
		{
			v = ((v & 0x00000000FFFFFFFFull) << 32) | ((v & 0xFFFFFFFF00000000ull) >> 32);
			v = ((v & 0x0000FFFF0000FFFFull) << 16) | ((v & 0xFFFF0000FFFF0000ull) >> 16);
			v = ((v & 0x00FF00FF00FF00FFull) <<  8) | ((v & 0xFF00FF00FF00FF00ull) >>  8);
			return v;
		}
		static inline uint32_t byteswap32(uint32_t v) noexcept
		{
			return (((v >> 24) & 0xff) | ((v >> 8) & 0xff00) | ((v << 8) & 0xff0000) | ((v << 24) & 0xff000000));
		}
	#endif
#endif


namespace wy
{
namespace internal
{
	// 128bit multiply function
#if WYHASH_32BIT_MUM
	static inline uint64_t _wyrot(uint64_t x) { return (x >> 32) | (x << 32); }
#endif
	static inline void _wymum(uint64_t* A, uint64_t* B) noexcept {
#if WYHASH_32BIT_MUM
		uint64_t hh = (*A >> 32) * (*B >> 32), hl = (*A >> 32) * (uint32_t)*B, lh = (uint32_t)*A * (*B >> 32), ll = (uint64_t)(uint32_t)*A * (uint32_t)*B;
	#if WYHASH_CONDOM > 1
		* A ^= _wyrot(hl) ^ hh; *B ^= _wyrot(lh) ^ ll;
	#else
		* A = _wyrot(hl) ^ hh; *B = _wyrot(lh) ^ ll;
	#endif
#elif defined(__SIZEOF_INT128__)
		__uint128_t r = *A; r *= *B;
	#if(WYHASH_CONDOM>1)
		* A ^= (uint64_t)r; *B ^= (uint64_t)(r >> 64);
	#else
		* A = (uint64_t)r; *B = (uint64_t)(r >> 64);
	#endif
#elif defined(_MSC_VER) && defined(_M_X64)
	#if WYHASH_CONDOM > 1
		uint64_t  a, b;
		a = _umul128(*A, *B, &b);
		*A ^= a;  *B ^= b;
	#else
		* A = _umul128(*A, *B, B);
	#endif
#else
		uint64_t ha = *A >> 32, hb = *B >> 32, la = (uint32_t)*A, lb = (uint32_t)*B, hi, lo;
		uint64_t rh = ha * hb, rm0 = ha * lb, rm1 = hb * la, rl = la * lb, t = rl + (rm0 << 32), c = t < rl;
		lo = t + (rm1 << 32); c += lo < t; hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
	#if WYHASH_CONDOM > 1
		* A ^= lo;  *B ^= hi;
	#else
		* A = lo;  *B = hi;
	#endif
#endif
	}

	// multiply and xor mix function, aka MUM
	static inline uint64_t _wymix(uint64_t A, uint64_t B) noexcept { _wymum(&A, &B); return A ^ B; }

	// read functions
#if WYHASH_LITTLE_ENDIAN
	static inline uint64_t _wyr8(const uint8_t* p) noexcept { uint64_t v; memcpy(&v, p, 8); return v; }
	static inline uint64_t _wyr4(const uint8_t* p) noexcept { uint32_t v; memcpy(&v, p, 4); return v; }
#else
	static inline uint64_t _wyr8(const uint8_t* p) noexcept { uint64_t v; memcpy(&v, p, 8); return byteswap64(v); }
	static inline uint64_t _wyr4(const uint8_t* p) noexcept { uint32_t v; memcpy(&v, p, 4); return byteswap32(v); }
#endif
	static inline uint64_t _wyr3(const uint8_t* p, size_t k) noexcept { return (((uint64_t)p[0]) << 16) | (((uint64_t)p[k >> 1]) << 8) | p[k - 1]; }

	// A useful 64bit-64bit mix function to produce deterministic pseudo random numbers that can pass BigCrush and PractRand
	static inline uint64_t wyhash64(uint64_t A, uint64_t B) noexcept { A ^= 0xa0761d6478bd642full; B ^= 0xe7037ed1a0b428dbull; _wymum(&A, &B); return _wymix(A ^ 0xa0761d6478bd642full, B ^ 0xe7037ed1a0b428dbull); }

	// The wyrand PRNG that pass BigCrush and PractRand
	static inline uint64_t wyrand(uint64_t* seed) noexcept { *seed += 0xa0761d6478bd642full; return _wymix(*seed, *seed ^ 0xe7037ed1a0b428dbull); }

#if !WYHASH_32BIT_MUM
	// fast range integer random number generation on [0,k) credit to Daniel Lemire. May not work when WYHASH_32BIT_MUM=1. It can be combined with wyrand, wyhash64 or wyhash.
	static inline uint64_t wy2u0k(uint64_t r, uint64_t k) noexcept { _wymum(&r, &k); return k; }
#endif
}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


namespace wy {

	/// <summary>
	/// Pseudo random numbers generator using WYRAND
	/// </summary>
	struct rand {
		uint64_t state; // Only value needed to generate pseudo-random numbers

		////////////////////////////////////////////////////////////////////////////////////
		// UniformRandomBitGenerator requirenment
		////////////////////////////////////////////////////////////////////////////////////
		using result_type = uint64_t; // is an unsigned integer type
		/// <summary>
		/// Returns the smallest value that operator() may return.
		/// The value is strictly less than max().
		/// The function must be constexpr.
		/// </summary>
		static constexpr result_type min() { return 0; }
		/// <summary>
		/// Returns the largest value that operator() may return.
		/// The value is strictly greater than min().
		/// The function must be constexpr.
		/// </summary>
		static constexpr result_type max() { return UINT64_MAX; }
		/// <summary>
		/// Returns a random value in the closed interval [0, UINT64_MAX].
		/// </summary>
		/// <returns>A new 64-bit pseudo-random value</returns>
		inline uint64_t operator()() noexcept
		{
			return internal::wyrand(&state);
		}
		////////////////////////////////////////////////////////////////////////////////////

		/// <summary>
		/// Construct a pseudo-random generator with a random seed
		/// </summary>
		rand() noexcept
		{
			// Seed with a real random value, if available
			std::random_device rd;
			state = static_cast<uint64_t>(rd()) | static_cast<uint64_t>(rd()) << 32;
		}
		/// <summary>
		/// Construct a pseudo-random generator with a given seed
		/// </summary>
		/// <param name="seed">The only value needed to generate the same sequence of pseudo-random numbers</param>
		rand(uint64_t seed) noexcept : state(seed)
		{}

		/// <summary>
		/// Generate a random value from the uniform distribution [0,1)
		/// </summary>
		/// <returns>The random value</returns>
		inline double uniform_dist() noexcept
		{
			uint64_t r = operator()();
			// code taken from 'wyhash.h::wy2u01(uint64_t r)'
			constexpr double _wynorm = 1.0 / (1ull << 52);
			return (r >> 12) * _wynorm;
		}

		/// <summary>
		/// Generate a random value from the uniform distribution [min_value, max_value)
		/// </summary>
		/// <param name="min_value">The minimum value (inclusive)</param>
		/// <param name="max_value">The maximum value (exclusive)</param>
		/// <returns>The random value</returns>
		inline double uniform_dist(double min_value, double max_value) noexcept
		{
			assert(max_value > min_value);

			return uniform_dist() * (max_value - min_value) + min_value;
		}

#if !WYHASH_32BIT_MUM
		/// <summary>
		/// Fast generation of a random value from the uniform distribution [0, max_value)
		/// </summary>
		/// <param name="max_value">The maximum value (exclusive)</param>
		/// <returns>The random value</returns>
		inline uint64_t uniform_dist(uint64_t max_value) noexcept
		{
			return internal::wy2u0k(operator()(), max_value);
		}
#endif

		/// <summary>
		/// Generate a random value from APPROXIMATE Gaussian distribution with mean=0 and std=1
		/// </summary>
		/// <returns>The random value</returns>
		inline double gaussian_dist() noexcept
		{
			uint64_t r = operator()();
			// code taken from 'wyhash.h::wy2gau(uint64_t r)'
			constexpr double _wynorm = 1.0 / (1ull << 20);
			return ((r & 0x1fffff) + ((r >> 21) & 0x1fffff) + ((r >> 42) & 0x1fffff)) * _wynorm - 3.0;
		}

		/// <summary>
		/// Generate a random value from APPROXIMATE Gaussian distribution with mean and std
		/// </summary>
		/// <param name="mean">The Gaussian mean</param>
		/// <param name="std">The Gaussian Standard Deviation</param>
		/// <returns>The random value</returns>
		inline double gaussian_dist(double mean, double std) noexcept
		{
			assert(std > 0);

			return gaussian_dist() * std + mean;
		}

		/// <summary>
		/// Generate a random stream of bytes.
		/// </summary>
		/// <typeparam name="T">The type of elements on the vector to fill with random data</typeparam>
		/// <param name="size">The number of elements of the vector to generate</param>
		/// <returns>A vector of random elements</returns>
		template<class T=uint8_t> inline std::vector<T> generate_stream(size_t size) noexcept
		{
			std::vector<T> result;
			generate_stream<T>(result, size);
			return result;
		}

		/// <summary>
		/// Generate a random stream of bytes.
		/// </summary>
		/// <typeparam name="T">The type of elements on the vector to fill with random data</typeparam>
		/// <param name="vec">out: A vector of random elements</param>
		/// <param name="size">The number of elements of the vector to generate</param>
		template<class T=uint8_t> void generate_stream(std::vector<T>& vec, size_t size) noexcept
		{
			size_t sizeOf64 = (size * sizeof(T) + sizeof(uint64_t) - 1) / sizeof(uint64_t); // The number of 64-bits numbers to generate

			// Create the memory on the vector
			vec.resize((sizeOf64 * sizeof(uint64_t) + sizeof(T) - 1) / sizeof(T));
			uint8_t* dataPtr = reinterpret_cast<uint8_t*>(vec.data());

			// Generate random values
			for (size_t i = 0; i < sizeOf64; i++)
			{
#if WYHASH_LITTLE_ENDIAN
				uint64_t val = operator()();
#else
				uint64_t val = byteswap64(operator()());
#endif
				memcpy(dataPtr + i * sizeof(uint64_t), &val, sizeof(uint64_t));
			}

			// Final size
			vec.resize(size);
		}
	};

	/// <summary>
	/// Internal implementations
	/// </summary>
	namespace internal {
		/// <summary>
		/// Hash base class
		/// </summary>
		struct hash_imp
		{
			uint64_t secret[4];// salted: We use dynamic secret to avoid intended attacks.

			/// <summary>
			/// Create a wyhasher with default secret
			/// </summary>
			hash_imp() noexcept : secret{ 0xa0761d6478bd642full, 0xe7037ed1a0b428dbull, 0x8ebc6af09c88c6e3ull, 0x589965cc75374cc3ull }// the default secret parameters
			{}
			/// <summary>
			/// Create a wyhasher with secret generated from a seed
			/// </summary>
			/// <param name="seed">The seed to generate the secret from</param>
			hash_imp(uint64_t seed) noexcept
			{
				// make your own secret
				// code taken from 'wyhash.h::make_secret(seed, secret)'
				uint8_t c[] = { 15, 23, 27, 29, 30, 39, 43, 45, 46, 51, 53, 54, 57, 58, 60, 71, 75, 77, 78, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 113, 114, 116, 120, 135, 139, 141, 142, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 177, 178, 180, 184, 195, 197, 198, 201, 202, 204, 209, 210, 212, 216, 225, 226, 228, 232, 240 };
				for (size_t i = 0; i < 4; i++) {
					uint8_t ok;
					do {
						ok = 1; secret[i] = 0;
						for (size_t j = 0; j < 64; j += 8) secret[i] |= ((uint64_t)c[wyrand(&seed) % sizeof(c)]) << j;
						if (secret[i] % 2 == 0) { ok = 0; continue; }
						for (size_t j = 0; j < i; j++) {
#if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
							if (__builtin_popcountll(secret[j] ^ secret[i]) != 32) { ok = 0; break; }
#elif defined(_MSC_VER) && defined(_M_X64)
							if (_mm_popcnt_u64(secret[j] ^ secret[i]) != 32) { ok = 0; break; }
#else
							// manual popcount
							uint64_t x = secret[j] ^ secret[i];
							x -= (x >> 1) & 0x5555555555555555;
							x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
							x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f;
							x = (x * 0x0101010101010101) >> 56;
							if (x != 32) { ok = 0; break; }
#endif
						}
					} while (!ok);
				}
			}
			/// <summary>
			/// Create a wyhasher with a specific secret
			/// </summary>
			/// <param name="secret">The secret to use</param>
			hash_imp(const uint64_t psecret[4]) noexcept
			{
				memcpy(secret, psecret, sizeof(secret));
			}

			/// <summary>
			/// Hash general data
			/// </summary>
			/// <param name="data">The data to hash</param>
			/// <param name="len">The size of the data</param>
			/// <returns>A 64-bits hash</returns>
			inline uint64_t wyhash(const uint8_t* data, size_t len) const noexcept
			{
				// code taken from 'wyhash.h::wyhash(const void* key, size_t len, uint64_t seed, const uint64_t * secret)' with seed=0
				const uint8_t* p = (const uint8_t*)data;
				uint64_t seed = secret[0], a, b;
				if (_likely_(len <= 16)) {
					if (_likely_(len >= 4)) { a = (_wyr4(p) << 32) | _wyr4(p + ((len >> 3) << 2)); b = (_wyr4(p + len - 4) << 32) | _wyr4(p + len - 4 - ((len >> 3) << 2)); }
					else if (_likely_(len > 0)) { a = _wyr3(p, len); b = 0; }
					else a = b = 0;
				}
				else {
					size_t i = len;
					if (_unlikely_(i > 48)) {
						uint64_t see1 = seed, see2 = seed;
						do {
							seed = _wymix(_wyr8(p) ^ secret[1], _wyr8(p + 8) ^ seed);
							see1 = _wymix(_wyr8(p + 16) ^ secret[2], _wyr8(p + 24) ^ see1);
							see2 = _wymix(_wyr8(p + 32) ^ secret[3], _wyr8(p + 40) ^ see2);
							p += 48; i -= 48;
						} while (_likely_(i > 48));
						seed ^= see1 ^ see2;
					}
					while (_unlikely_(i > 16)) { seed = _wymix(_wyr8(p) ^ secret[1], _wyr8(p + 8) ^ seed);  i -= 16; p += 16; }
					a = _wyr8(p + i - 16);  b = _wyr8(p + i - 8);
				}
				return _wymix(secret[1] ^ len, _wymix(a ^ secret[1], b ^ seed));

			}
			/// <summary>
			/// Hash a 64-bit number
			/// </summary>
			/// <param name="number">The number to hash</param>
			/// <returns>A 64-bits hash</returns>
			inline uint64_t wyhash(uint64_t number) const noexcept
			{
				return internal::wyhash64(number, secret[0]);
			}
		};

		/// <summary>
		/// Hash base class for string types
		/// </summary>
		/// <typeparam name="STRING_TYPE">The type of the string, ex: std::string, std::wstring, ...</typeparam>
		template<class STRING_TYPE> struct hash_string_base : private hash_imp
		{
			using hash_imp::hash_imp;// Inherit constructors
			inline uint64_t operator()(const STRING_TYPE& elem) const noexcept
			{
				return hash_imp::wyhash(reinterpret_cast<const uint8_t*>(elem.data()), sizeof(typename STRING_TYPE::value_type) * elem.size());
			}

			inline uint64_t operator()(const typename STRING_TYPE::value_type* data, const size_t size) const noexcept
			{
				return hash_imp::wyhash(reinterpret_cast<const uint8_t*>(data), sizeof(typename STRING_TYPE::value_type) * size);
			}
		};
	};

	/// <summary>
	/// Common wyhash for general use
	/// </summary>
	/// <typeparam name="T">Type of the element to hash</typeparam>
	template<class T> struct hash : private internal::hash_imp
	{
		using hash_imp::hash_imp;// Inherit constructors

		/// <summary>
		/// Hash a general type
		/// </summary>
		/// <param name="elem">The element to hash</param>
		/// <returns>A 64-bits hash</returns>
		inline uint64_t operator()(const T& elem) const noexcept
		{
			static_assert(sizeof(T) > 0, "Type to hash T should have variables");
			return hash_imp::wyhash(reinterpret_cast<const uint8_t*>(&elem), sizeof(T));
		}
	};
	/// <summary>
	/// Partial specialization for pointer
	/// </summary>
	/// <typeparam name="T">Type of elements</typeparam>
	template<class T> struct hash<T*> : private internal::hash_imp
	{
		using hash_imp::hash_imp;// Inherit constructors
		inline uint64_t operator()(const T* elem) const noexcept
		{
			static_assert(sizeof(T) > 0, "Type to hash T should have variables");
			return hash_imp::wyhash(reinterpret_cast<const uint8_t*>(elem), sizeof(T));
		}
	};
	
	// Partial specializations: number
	template<> struct hash<uint64_t> : private internal::hash_imp
	{
		using hash_imp::hash_imp;// Inherit constructors
		inline uint64_t operator()(uint64_t number) const noexcept
		{
			return hash_imp::wyhash(number);
		}
	};
	template<> struct hash<int64_t> : private internal::hash_imp
	{
		using hash_imp::hash_imp;// Inherit constructors
		inline uint64_t operator()(int64_t number) const noexcept
		{
			return hash_imp::wyhash(number);
		}
	};

	// Partial specializations: std::vector
	//template<class T> struct hash<std::vector<T>> : public internal::hash_string_base<std::vector<T>>
	//{
	//	using hash_string_base::hash_string_base;// Inherit constructors
	//};

	// C strings
	template<> struct hash<char*> : private internal::hash_imp
	{
		using hash_imp::hash_imp;// Inherit constructors
		inline uint64_t operator()(const char* data) const noexcept
		{
			return hash_imp::wyhash(reinterpret_cast<const uint8_t*>(data), strlen(data));
		}
	};
	template<> struct hash<const char*> : private internal::hash_imp
	{
		using hash_imp::hash_imp;// Inherit constructors
		inline uint64_t operator()(const char* data) const noexcept
		{
			return hash_imp::wyhash(reinterpret_cast<const uint8_t*>(data), strlen(data));
		}
	};

	// Partial specializations: std::string variants
	template<> struct hash<std::string> : public internal::hash_string_base<std::string>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
	template<> struct hash<std::wstring> : public internal::hash_string_base<std::wstring>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
	template<> struct hash<std::u16string> : public internal::hash_string_base<std::u16string>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
	template<> struct hash<std::u32string> : public internal::hash_string_base<std::u32string>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
	
	// std::string_view variants
#if __cpp_lib_string_view
	template<> struct hash<std::string_view> : public internal::hash_string_base<std::string_view>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
	template<> struct hash<std::wstring_view> : public internal::hash_string_base<std::wstring_view>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
	template<> struct hash<std::u16string_view> : public internal::hash_string_base<std::u16string_view>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
	template<> struct hash<std::u32string_view> : public internal::hash_string_base<std::u32string_view>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
#endif

	// std::pmr::string variants
#if __cpp_lib_polymorphic_allocator
	template<> struct hash<std::pmr::string> : public internal::hash_string_base<std::pmr::string>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
	template<> struct hash<std::pmr::wstring> : public internal::hash_string_base<std::pmr::wstring>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
	template<> struct hash<std::pmr::u16string> : public internal::hash_string_base<std::pmr::u16string>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
	template<> struct hash<std::pmr::u32string> : public internal::hash_string_base<std::pmr::u32string>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
#endif

	// char8_t string variants
#if __cpp_char8_t
	template<> struct hash<std::u8string> : public internal::hash_string_base<std::u8string>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
	template<> struct hash<std::u8string_view> : public internal::hash_string_base<std::u8string_view>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
	template<> struct hash<std::pmr::u8string> : public internal::hash_string_base<std::pmr::u8string>
	{
		using hash_string_base::hash_string_base;// Inherit constructors
	};
#endif
};

#endif // WYHASH_WYHASH_HPP
