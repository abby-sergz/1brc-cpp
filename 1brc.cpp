#include <assert.h>
#include <algorithm>
#include <array>
#include <bit>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <functional>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string_view>
#include <string>
#include <thread>
#include <unordered_map>

#ifdef WIN32

#define NOMINMAX
#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <intrin.h>

static void dump_times()
{
	HANDLE hProcess = GetCurrentProcess();

	// Variables to store timing information
	FILETIME creationTime, exitTime, kernelTime, userTime;

	// Get process times
	if (GetProcessTimes(hProcess, &creationTime, &exitTime, &kernelTime, &userTime)) {
		// Convert FILETIME to ULARGE_INTEGER for easy manipulation
		ULARGE_INTEGER kernelTimeUL, userTimeUL;
		kernelTimeUL.LowPart = kernelTime.dwLowDateTime;
		kernelTimeUL.HighPart = kernelTime.dwHighDateTime;
		userTimeUL.LowPart = userTime.dwLowDateTime;
		userTimeUL.HighPart = userTime.dwHighDateTime;

		// Print the times in milliseconds
		std::cout << "Kernel Time: " << kernelTimeUL.QuadPart / 10000 << " ms\n";
		std::cout << "User Time: " << userTimeUL.QuadPart / 10000 << " ms\n";
	}
	else {
		// Handle error
		std::cerr << "Error: " << GetLastError() << std::endl;
	}
}

class ElapsedTimer {
public:
	ElapsedTimer() {
		QueryPerformanceCounter(&start);
	}

	uint64_t elapsed_msec() const {
		LARGE_INTEGER end;
		QueryPerformanceCounter(&end);
		return (end.QuadPart - start.QuadPart) * 1000llu / frequency.QuadPart;
	}

	void dump(const std::string_view& prefix, std::ostream& os = std::cout) const
	{
		os << prefix << " " << elapsed_msec() << " msec\n";
	}

private:
	LARGE_INTEGER start;
	static LARGE_INTEGER frequency;
};

LARGE_INTEGER ElapsedTimer::frequency = []() {
	LARGE_INTEGER f;
	QueryPerformanceFrequency(&f);
	return f;
}();

class MMFile
{
public:
	explicit MMFile(const std::string_view& file_path)
	{
		m_h_file = CreateFileA(file_path.data(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
		if (m_h_file == INVALID_HANDLE_VALUE) {
			std::ostringstream oss;
			oss << "Error opening file: " << GetLastError();
			throw std::runtime_error(oss.str());
		}
		// One may not map zero length files
		if (!GetFileSizeEx(m_h_file, &m_size) || m_size.QuadPart == 0) {
			std::ostringstream oss;
			oss << "Error getting file size: " << GetLastError();
			throw std::runtime_error(oss.str());
		}
		m_h_mmfile = CreateFileMappingA(m_h_file, NULL, PAGE_READONLY, 0, 0, NULL);
		if (m_h_mmfile == NULL) {
			std::ostringstream oss;
			oss << "Error creating memory mapped file: " << GetLastError();
			throw std::runtime_error(oss.str());
		}
		m_h_mmview = MapViewOfFile(m_h_mmfile, FILE_MAP_READ, 0, 0, 0);
		if (m_h_mmview == NULL) {
			std::ostringstream oss;
			oss << "Error mapping memory mapped view: " << GetLastError();
			throw std::runtime_error(oss.str());
		}
	}
	~MMFile()
	{
		if (m_h_mmview)
		{
			UnmapViewOfFile(m_h_mmview);
		}
		if (m_h_mmfile)
		{
			CloseHandle(m_h_mmfile);
		}
		if (m_h_file)
		{
			CloseHandle(m_h_file);
		}
	}

	const char* data() const
	{
		return static_cast<const char*>(m_h_mmview);
	}

	size_t size() const
	{
		static_assert(sizeof(size_t) == sizeof(m_size.QuadPart));
		return m_size.QuadPart;
	}
private:
	HANDLE m_h_file = INVALID_HANDLE_VALUE;
	HANDLE m_h_mmfile = nullptr;
	LPVOID m_h_mmview = nullptr;
	LARGE_INTEGER m_size{};
};
#else

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>

class MMFile
{
public:
	explicit MMFile(const std::string_view& file_path)
	{
		m_fd = open(file_path.data(), O_RDONLY);
		if (m_fd == -1)
		{
			std::ostringstream oss;
			oss << "Error opening file: " << errno << strerror(errno);
			throw std::runtime_error(oss.str());
		}
		struct stat file_stat {};
		if (fstat(m_fd, &file_stat) == -1)
		{
			std::ostringstream oss;
			oss << "Error getting file size: " << errno << strerror(errno);
			throw std::runtime_error(oss.str());
		}
		
		m_size = file_stat.st_size;

		m_h_mmview = mmap(nullptr, m_size, PROT_READ, MAP_SHARED, m_fd, 0);
		if (m_h_mmview == MAP_FAILED) {
			std::ostringstream oss;
			oss << "mapping memory mapped view: " << errno << strerror(errno);
			throw std::runtime_error(oss.str());
		}
	}
	~MMFile()
	{
		if (m_h_mmview != MAP_FAILED)
		{
			munmap(m_h_mmview, m_size);
		}
		if (m_fd != -1)
		{
			::close(m_fd);
		}
	}
	const char* data() const
	{
		return static_cast<const char*>(m_h_mmview);
	}

	size_t size() const
	{
		return m_size;
	}
private:
	int m_fd;
	void* m_h_mmview = MAP_FAILED;
	size_t m_size = 0;
};
#endif

#define USE_STD_UMAP 0
// 1 is for AVX256
#define USE_SIMD 2
#define SINGLE_THREADED_AGG 0
#define NOT_USE_MM_FILE 1

namespace sergz_1brc
{
	struct entry_type
	{
		typedef int16_t value_type;
		// No much thoughts about the input values, but it should be enough.
		// Who knows what temperatures are there, int8_t seems not enough.
		// 1 000 000 000 definitely fits into uint32_t, but might be adjusted.
		// Apparently max sum is 2^16 * 2^32 = 2^48, what should fit into 2^64.
		// 
		// The generating code rounds measurements to deciman fraction, therefore we can multiply it by 10 here in order to avoid slow floating point numbers
		int64_t sum = 0; // multiplied by 10
		uint32_t count = 0;
		value_type min;
		value_type max;
		float avg;

		explicit entry_type(value_type value = 0)
		{
			sum = value;
			count = 1;
			min = value;
			max = value;
		}

		void update(value_type value)
		{
			sum += value;
			++count;
			if (value < min)
			{
				min = value;
			}
			if (value > max)
			{
				max = value;
			}
		}

		void update(entry_type value)
		{
			sum += value.sum;
			count += value.count;
			if (value.min < min)
			{
				min = value.min;
			}
			if (value.max > max)
			{
				max = value.max;
			}
		}
	};

	// It seems the majority of keys actually are a subject to short string optimization, otherwise, we need something in addition.
	// 
#if USE_STD_UMAP
	class my_simple_hash
	{
	public:
		explicit my_simple_hash(size_t capacity = 1000)
		{
			m_map.reserve(capacity);
		}
		template<typename T, typename V>
		void upsert_fn(const T& key, V&& value)
		{
			auto [ii_agg, is_inserted] = m_map.insert(std::make_pair(key, entry_type(value)));
			if (!is_inserted)
			{
				ii_agg->second.update(value);
			}
		}
		const entry_type& operator[](const std::string_view& key) const
		{
			std::string k(key);
			return m_map.at(k);
		}
		template<typename Fn>
		void for_each(Fn&& fn)
		{
			for (auto& [key, value] : m_map)
			{
				fn(key, value);
			}
		}
		size_t size() const
		{
			return m_map.size();
		}
	private:
		std::unordered_map<std::string, entry_type> m_map;
	};
#else
#ifdef _DEBUG
	// take a look at the distribution. If it's bad, then change something.
	static std::map<size_t, size_t> depth_stats;
#endif
	// there is no growth, data structures are inlined, etc.
	class my_simple_hash
	{
		template<typename T>
		static size_t fnv_1a(const T& key)
		{
			// FNV-1a hash function
			size_t result = 2166136261;
			for (size_t i = 0; i < key.size(); ++i)
				result = (result ^ key[i]) * 16777619;
			return result;
		}
		struct Bucket
		{
			std::string key;
			entry_type value;
			bool is_used = false;
		};
	public:
		// load factor is already here
		explicit my_simple_hash(size_t capacity = 1000)
		{
			if (!capacity)
			{
				return;
			}
			size_t bucket_num = 1;
			while (bucket_num < capacity)
				bucket_num <<= 1;
			m_buckets.resize(bucket_num);
		}

		my_simple_hash(my_simple_hash&& src) noexcept
			: m_buckets(std::move(src.m_buckets))
		{
			std::swap(m_size, src.m_size);
			std::swap(m_max_depth, src.m_max_depth);
		}

		my_simple_hash& operator=(my_simple_hash&& src) noexcept
		{
			m_buckets = std::move(src.m_buckets);
			std::swap(m_size, src.m_size);
			std::swap(m_max_depth, src.m_max_depth);
			return *this;
		}


		template<typename T, typename V>
		void upsert_fn(const T& key, V value)
		{
			size_t hash = fnv_1a(key);
			// pos = (hash + SUM[1 to i]) mod bucketCount, where SUM = i*(i + 1)/2, so basically it's a cheap quadratic probing
			// but try linear
			for (size_t i = 0; ; ++i)
			{
				if (i > m_max_depth)
				{
					m_max_depth = i;
				}
				// hash % bucket_num, however since bucket_num is 2^n
				// (hash & bucket_num - 1) is equivalent to it, but hopefully is faster
				Bucket& bucket = m_buckets[hash & (m_buckets.size() - 1)];
				if (!bucket.is_used)
				{
					bucket.key = key;
					bucket.value = entry_type(value);
					bucket.is_used = true;
					++m_size;
					if (m_size == m_buckets.size())
					{
						// Although it's not a failed state yet, it's definitely something bad.
						// For this particular case, let's fail already.
						throw std::runtime_error("Filled the last element to the hash table");
					}
#ifdef _DEBUG
					++depth_stats[i];
#endif
					return;
				}
				else if (bucket.key == key)
				{
					bucket.value.update(value);
					return;
				}
				hash += i;
				
			}
		}

		const entry_type& operator[](const std::string_view &key) const
		{
			size_t hash = fnv_1a(key);
			// pos = (hash + SUM[1 to i]) mod bucketCount, where SUM = i*(i + 1)/2, so basically it's a cheap quadratic probing
			// but try linear
			for (size_t i = 1; i <= m_max_depth; ++i)
			{
				// h % bucket_num, however since bucket_num is 2^n
				// (h & bucket_num - 1) is equivalent to it, but hopefully is faster
				const Bucket& bucket = m_buckets[hash & (m_buckets.size() - 1)];
				if (bucket.is_used && bucket.key == key)
				{
					return bucket.value;
				}
				hash += i;
			}
			throw std::runtime_error(std::string("Not found key: ").append(key));
		}

		template<typename Fn>
		void for_each(Fn&& fn)
		{
			for (auto& bucket : m_buckets)
			{
				if (bucket.is_used)
					fn(bucket.key, bucket.value);
			}
		}
		template<typename Fn>
		void for_each(Fn&& fn) const
		{
			for (auto& bucket : m_buckets)
			{
				if (bucket.is_used)
					fn(bucket.key, bucket.value);
			}
		}
		size_t size() const
		{
			return m_size;
		}
	private:
		size_t m_size = 0;
		size_t m_max_depth = 0;
		std::vector<Bucket> m_buckets;
	};
#endif

	// It provides a thread pool with a FIFO queue.
	template<typename Payload>
	class async_processor
	{
		template<typename Q>
		class circular_buffer {
			// In theory the implementation of std::deque varies and has some optimizations, like using
			// chunks of continuous memory under the hood. We don't remove elements from the middle of
			// the queue, so there will be no case when there is a single element in a chunk, which keeps
			// the chunk from being freed and there are many chunks of unknown size.
			// However, std::deque still causes an uncontrolled memory growth.
		public: // std::deque compatible
			bool push_back(Q&& value) {
				grow();
				m_tail = next(m_tail);
				m_buffer[m_tail] = std::move(value);
				++m_size;
				return true;
			}

			void pop_front()
			{
				if (m_size == 0)
				{
					// it should never happen
					return;
				}
				--m_size;
				if (m_size == 0)
				{
					m_head = 0;
					m_tail = std::numeric_limits<uint32_t>::max();
				}
				else
					m_head = next(m_head);
				shrink();
			}

			Q& front() {
				return m_buffer[m_head];
			}

			bool empty() const {
				return m_size == 0;
			}
		private:
			void grow()
			{
				if (m_size != m_buffer.size())
					return;
				if (m_head < m_tail)
				{
					m_buffer.resize(std::max<uint32_t>(1, m_size) * 2);
				}
				else
				{
					std::vector<Q> nbuffer(std::max<uint32_t>(1, m_size) * 2);
					for (uint32_t pos = 0; pos < m_size; ++pos) {
						std::swap(nbuffer[pos], m_buffer[(m_head + pos) % m_buffer.size()]);
					}
					std::swap(nbuffer, m_buffer);
					m_head = 0;
					m_tail = m_size - 1;
				}
			}
			void shrink()
			{
				if (m_size * 4 > m_buffer.capacity())
					return;
				std::vector<Q> nbuffer(std::max<uint32_t>(1, m_size) * 2);
				for (uint32_t pos = 0; pos < m_size; ++pos) {
					std::swap(nbuffer[pos], m_buffer[(m_head + pos) % m_buffer.size()]);
				}
				std::swap(nbuffer, m_buffer);
				m_head = 0;
				m_tail = m_size - 1;
			}
			uint32_t next(uint32_t i) const
			{
				// it's fine to overflow here because the type is unsigned
				return (i + 1) % m_buffer.size();
			}
		private:
			std::vector<Q> m_buffer;
			uint32_t m_head = 0;
			uint32_t m_tail = std::numeric_limits<uint32_t>::max();
			uint32_t m_size = 0;
		};

		enum class exit_codes : uint8_t
		{
			none,
			wait_empty_queue,
		};
	public:
		explicit async_processor(uint16_t threads_number,
			const std::function<void(Payload&&)>& handler)
			: m_handler(handler)
		{
			if (!m_handler)
				throw std::logic_error("Payload handler for async_processor is empty");
			for (uint16_t i = 0; i < threads_number; ++i)
			{
				m_threads.emplace_back(&async_processor::thread_func, this);
			}
		}
		~async_processor()
		{
			{
				std::lock_guard<std::mutex> lock(m_mutex);
				m_exit = exit_codes::wait_empty_queue;
			}
			m_cv.notify_all();
			for (auto& thread : m_threads)
			{
				if (thread.joinable())
				{
					thread.join();
				}
			}
		}
		void send(Payload&& payload)
		{
			if (m_threads.empty())
			{
				return call_handler(std::move(payload));
			}
			{
				std::lock_guard<std::mutex> lock(m_mutex);
				m_queue.push_back(std::move(payload));
			}
			m_cv.notify_one();
		}
	private:
		void call_handler(Payload&& payload)
		{
			// it it throws then crash here
			m_handler(std::move(payload));
		}
		void thread_func()
		{
			while (true)
			{
				// It has to be default constructible, but on the other hand, no manual unlock/lock.
				Payload current_element{};
				{
					std::unique_lock<std::mutex> lock(m_mutex);
					while (m_exit == exit_codes::none && m_queue.empty())
					{
						m_cv.wait(lock);
					}
					if (m_queue.empty() && m_exit == exit_codes::wait_empty_queue)
						return;
					current_element = std::move(m_queue.front());
					m_queue.pop_front();
				}
				call_handler(std::move(current_element));
			}
		}
	private:
		std::function<void(Payload&&)> m_handler;
		std::mutex m_mutex;
		std::condition_variable m_cv;
		circular_buffer<Payload> m_queue;
		std::vector<std::thread> m_threads;
		exit_codes m_exit = exit_codes::none;
	};


	enum class field_type
	{
		place_name, value_sign, value, carried_place_name
	};

	struct reset_next_line_entry_data
	{
		field_type curr_field = field_type::place_name;
		bool is_neg = false;
		int16_t value = 0;
		bool is_complete_line = false;
	};
	struct parse_entry_state : reset_next_line_entry_data
	{
		enum class use_simd
		{
			no_simd, avx256, avx512
		};
		std::string_view place_name;
		std::string carry_place_name;

		template<typename Fn>
		void parse_string(const char* data_begin, const char* data_end, Fn&& fn)
		{
			switch (available_simd_method)
			{
			case use_simd::avx256:
				parse_string_avx<use_simd::avx256>(data_begin, data_end, std::forward<Fn>(fn));
				break;
			case use_simd::avx512:
				parse_string_avx<use_simd::avx512>(data_begin, data_end, std::forward<Fn>(fn));
				break;
			case use_simd::no_simd: [[fallthrough]];
			default:
				parse_string(data_begin, data_end, fn, data_begin);
				break;
			}
		}

		template<use_simd simd_method>
		struct AVXSearch;

	private:
		template<typename Fn>
		void parse_string(const char* data_begin, const char* data_end, Fn&& fn, const char* data)
		{
			while (true)
			{
				if (data == data_end)
				{
					if (curr_field == field_type::place_name)
					{
						curr_field = field_type::carried_place_name;
						carry_place_name.assign(data_begin, data_end - data_begin);
					}
					else
					{
						carry_place_name.assign(place_name);
					}
					fn();
					break;
				}
				char c = *data;
				switch (curr_field)
				{
				case field_type::place_name:
					if (c == ';')
					{
						place_name = std::string_view(data_begin, data - data_begin);
						curr_field = field_type::value_sign;
					}
					break;
				case field_type::value_sign:
					curr_field = field_type::value;
					if (c == '-')
					{
						is_neg = true;
						break;
					}
					[[fallthrough]];
				case field_type::value:
					if (c == '\n')
					{
						if (is_neg)
						{
							value = -value;
						}
						is_complete_line = true;
						fn();
						static_cast<reset_next_line_entry_data&>(*this) = reset_next_line_entry_data{};
						carry_place_name.resize(0);
						data_begin = data + 1;
					}
					else if (c == '.' || c == '\r')
					{
					}
					else
					{
						value *= 10;
						value += c - '0';
					}
					break;
				case field_type::carried_place_name:
					if (c == ';')
					{
						carry_place_name.append(data_begin, data - data_begin);
						curr_field = field_type::value_sign;
					}
					break;
				default:
					;
				}
				++data;
			}
		}

		template<use_simd simd_method, typename Fn>
		void parse_string_avx(const char* data_begin, const char* data_end, Fn&& fn)
		{
			AVXSearch<simd_method> avx_code;
			const char* data = data_begin;
			const char* data_end_simd = data_end - AVXSearch<simd_method>::data_step;
			while (data < data_end_simd)
			{
				auto mask_special_symbol = avx_code.load_and_cmp(data);
				
				const char* wdata = data;
				while (mask_special_symbol != 0)
				{
#ifdef WIN32
					int special_symbol_pos = std::countr_zero(mask_special_symbol);
#else
					int special_symbol_pos = __builtin_ctzll(mask_special_symbol);
#endif
					const char* wdata_end = data + special_symbol_pos;

					if (curr_field == field_type::place_name)
					{
						assert(*wdata_end == ';');
						place_name = std::string_view(data_begin, wdata_end - data_begin);
						curr_field = field_type::value_sign;
						wdata = wdata_end + 1;
						mask_special_symbol &= ~(typename AVXSearch<simd_method>::mask_type(1) << special_symbol_pos);
						continue;
					}
					if (curr_field == field_type::value_sign)
					{
						curr_field = field_type::value;
						if (*wdata == '-')
						{
							is_neg = true;
							++wdata;
						}
					}
					if (curr_field == field_type::value)
					{
						assert(*wdata_end == '\n');
						while (wdata < wdata_end)
						{
							char c = *wdata;
							if (c == '.' || c == '\r')
							{
							}
							else
							{
								value *= 10;
								value += c - '0';
							}
							++wdata;
						}

						if (is_neg)
						{
							value = -value;
						}
						is_complete_line = true;
						fn();
						static_cast<reset_next_line_entry_data&>(*this) = reset_next_line_entry_data{};
						carry_place_name.resize(0);
						wdata = wdata_end + 1;
						data_begin = wdata;
						mask_special_symbol &= ~(typename AVXSearch<simd_method>::mask_type(1) << special_symbol_pos);
						continue;
					}
					
					if (curr_field == field_type::carried_place_name)
					{
						assert(*wdata_end == ';');
						carry_place_name.append(data_begin, wdata_end - data_begin);
						curr_field = field_type::value_sign;
						wdata = wdata_end + 1;
						mask_special_symbol &= ~(typename AVXSearch<simd_method>::mask_type(1) << special_symbol_pos);
						continue;
					}
				}

				data += AVXSearch<simd_method>::data_step;
				if (mask_special_symbol == 0 && wdata < data)
				{
					if (curr_field == field_type::value_sign)
					{
						curr_field = field_type::value;
						if (*wdata == '-')
						{
							is_neg = true;
							++wdata;
						}
					}
					if (curr_field == field_type::value)
					{
						while (wdata < data)
						{
							char c = *wdata;
							if (c == '.' || c == '\r')
							{
							}
							else
							{
								value *= 10;
								value += c - '0';
							}
							++wdata;
						}
					}
				}
			}
			parse_string(data_begin, data_end, fn, data);
		}

		static const use_simd available_simd_method;
	};
		template<>
		struct parse_entry_state::AVXSearch<parse_entry_state::use_simd::avx256>
		{
			static constexpr uint64_t data_step = 32;
			__m256i semicolon = _mm256_set1_epi8(';');
			__m256i newline = _mm256_set1_epi8('\n');
            typedef uint32_t mask_type;
			uint32_t load_and_cmp(const char* data)
			{
				__m256i simd_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
				__m256i simd_cmp_semicolon = _mm256_cmpeq_epi8(simd_data, semicolon);
				__m256i simd_cmp_newline = _mm256_cmpeq_epi8(simd_data, newline);

				uint32_t mask_semicolon = _mm256_movemask_epi8(simd_cmp_semicolon);
				uint32_t mask_newline = _mm256_movemask_epi8(simd_cmp_newline);
				// let's begin with something stupid and simple
				return mask_semicolon | mask_newline;
			}
		};

		template<>
		struct parse_entry_state::AVXSearch<parse_entry_state::use_simd::avx512>
		{
			static constexpr uint64_t data_step = 64;
			__m512i semicolon = _mm512_set1_epi8(';');
			__m512i newline = _mm512_set1_epi8('\n');
            typedef uint64_t mask_type;

			uint64_t load_and_cmp(const char* data)
			{
				__m512i simd_data = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data));
				__mmask64 mask_semicolon = _mm512_cmp_epu8_mask(simd_data, semicolon, _MM_CMPINT_EQ);
				__mmask64 mask_newline = _mm512_cmp_epu8_mask(simd_data, newline, _MM_CMPINT_EQ);
				// let's begin with something stupid and simple
				return mask_semicolon | mask_newline;
			}
		};


	const parse_entry_state::use_simd parse_entry_state::available_simd_method = []() {
		// there is no simple method (check a couple of bits) of figuring out of what is supported, so hard code it
		if constexpr (USE_SIMD == 2)
		{
			return use_simd::avx512;
		}

		if constexpr (USE_SIMD == 1)
		{
			return use_simd::avx256;
		}
		return use_simd::no_simd;
	}();
}

static int my_main(int argc, char* argv[])
{
	using namespace sergz_1brc;

	if (argc < 2) {
		return 1;
	}

	// Hopefully we will always fall into short string optimization.
	// https://worldpopulationreview.com/world-city-rankings/longest-city-names

	typedef my_simple_hash aggregates_type;

#if NOT_USE_MM_FILE || SINGLE_THREADED_AGG
	//*/
	constexpr size_t buffer_size = 4 * (1 << 20);
	/*/
	constexpr size_t buffer_size =  (1 << 10);
	//*/
#endif
	aggregates_type aggregates;

#if SINGLE_THREADED_AGG
	std::vector<char> buffer(buffer_size);
	{
		std::ifstream input_file(argv[1], std::ios::in | std::ios::binary);
		parse_entry_state pes{};
		pes.carry_place_name.reserve(4096);
		while (input_file.good())
		{
			if (input_file.read(buffer.data(), buffer_size).good() || input_file.eof())
			{
				auto data_size = input_file.gcount();
				const char* data = buffer.data();
				const char* data_end = buffer.data() + data_size;
				pes.parse_string(data, data_end, [&pes, &input_file, &aggregates]()
					{
						if (pes.is_complete_line || input_file.eof())
						{
							if (pes.carry_place_name.empty())
							{
								aggregates.upsert_fn(pes.place_name, pes.value);
							}
							else
							{
								aggregates.upsert_fn(pes.carry_place_name, pes.value);
							}
						}
					});
				
			}
		}
	}
#else
	std::mutex aggregates_mutex;
#if NOT_USE_MM_FILE
	struct SmallChunkToAggregate
	{
		// buffer always begins with a fresh string. If we cannot read the full string, then move its
		// beginning to the beginning of the buffer and continue.
		// It's always aligned to the data entries.
		// If memory mapped files is not an option and one can read faster than process than consider
		// returning of the buffer because memory de-/allocation is very slow.
		std::vector<char> buffer;
	};
	{
		async_processor<SmallChunkToAggregate> chunks_aggregator(static_cast<uint16_t>(std::max(std::min<unsigned int>(std::thread::hardware_concurrency(), std::numeric_limits<uint16_t>::max()), 5u) - 4u),
			[&aggregates, &aggregates_mutex](const auto& chunk)
			{
				aggregates_type chunk_aggregates;
				const auto& buffer = chunk.buffer;
				const char* data = buffer.data();
				parse_entry_state pes{};
				pes.parse_string(data, data + buffer.size(), [&pes, &chunk_aggregates]()
						{
							if (pes.is_complete_line)
							{
								if (pes.carry_place_name.empty())
								{
									chunk_aggregates.upsert_fn(pes.place_name, pes.value);
								}
								else
								{
									chunk_aggregates.upsert_fn(pes.carry_place_name, pes.value);
								}
							}
						});
				std::lock_guard<std::mutex> agg_lock(aggregates_mutex);
				// it's possible to not copy here but rather std::move, but it's not an issue. So, it's OK.
				chunk_aggregates.for_each([&aggregates](const std::string& key, const entry_type& entry)
					{
						aggregates.upsert_fn(key, entry);
					});
			});
		{
			std::ifstream input_file(argv[1], std::ios::in | std::ios::binary);
			std::vector<char> carry_buffer;
			while (input_file.good())
			{
				SmallChunkToAggregate data_chunk;
				data_chunk.buffer.resize(carry_buffer.size() + buffer_size);
				std::copy_n(carry_buffer.begin(), carry_buffer.size(), data_chunk.buffer.begin());
				if (input_file.read(data_chunk.buffer.data() + carry_buffer.size(), buffer_size).good() || input_file.eof())
				{
					auto data_size = input_file.gcount();
					assert(data_size > 0);
					const char* data_end = data_chunk.buffer.data() + carry_buffer.size() + data_size;
					const char* aligned_data_end = data_end - 1;
					while (*aligned_data_end != '\n')
					{
						--aligned_data_end;
					}
					++aligned_data_end;
					size_t carry_size = data_end - aligned_data_end;
					carry_buffer.resize(carry_size);
					std::copy_n(aligned_data_end, carry_size, carry_buffer.begin());
					data_chunk.buffer.resize(aligned_data_end - data_chunk.buffer.data());

					chunks_aggregator.send(std::move(data_chunk));
				}
			}
		}
	}
#else

// it's not worth it
#define USE_AGG_THREAD 0
#if USE_AGG_THREAD
	struct AggregatedChunk
	{
		aggregates_type agg;
		AggregatedChunk()
			: agg(0)
		{

		}
		AggregatedChunk(aggregates_type&& src)
			: agg(std::move(src))
		{
		}
	};
	async_processor<AggregatedChunk> chunks_aggregator(1u, [&aggregates](const auto& agg_chunk)
		{
			// it's possible to not copy here but rather std::move, but it's not an issue. So, it's OK.
			agg_chunk.agg.for_each([&aggregates](const std::string& key, const entry_type& entry)
				{
					aggregates.upsert_fn(key, entry);
				});
		});
#endif
	struct SmallChunkToAggregate
	{
		std::string_view buffer;
	};

	MMFile input_file(argv[1]);
	{
		//*/
		uint16_t threads_number = static_cast<uint16_t>(std::max(std::min<unsigned int>(std::thread::hardware_concurrency(), std::numeric_limits<uint16_t>::max()), 5u) - 2u);
		/*/
		uint16_t threads_number = 1;
		//*/
		async_processor<SmallChunkToAggregate> single_chunk_aggregator(threads_number, [
#if USE_AGG_THREAD
			&chunks_aggregator
#else
			&aggregates, &aggregates_mutex
#endif
		](const auto& chunk)
			{
				aggregates_type chunk_aggregates;
				const auto& buffer = chunk.buffer;
				const char* data = buffer.data();
				parse_entry_state pes{};
				pes.parse_string(data, data + buffer.size(), [&pes, &chunk_aggregates]()
						{
							if (pes.is_complete_line)
							{
								if (pes.carry_place_name.empty())
								{
									chunk_aggregates.upsert_fn(pes.place_name, pes.value);
								}
								else
								{
									chunk_aggregates.upsert_fn(pes.carry_place_name, pes.value);
								}
							}
						});
#if USE_AGG_THREAD
				chunks_aggregator.send({ std::move(chunk_aggregates) });
#else
				std::lock_guard<std::mutex> agg_lock(aggregates_mutex);
				chunk_aggregates.for_each([&aggregates](const std::string& key, const entry_type& entry)
					{
						aggregates.upsert_fn(key, entry);
					});
#endif

			});
		uint64_t block_size = (input_file.size() + threads_number - 1) / threads_number;
		const char* data_begin = input_file.data();
		const char* file_end = data_begin + input_file.size();
		while (data_begin < file_end)
		{
			const char* data_end = std::min(file_end, data_begin + block_size);
			const char* aligned_data_end = data_end - 1;
			while (*aligned_data_end != '\n')
			{
				--aligned_data_end;
			}
			++aligned_data_end;

			SmallChunkToAggregate data_chunk;
			data_chunk.buffer = std::string_view(data_begin, aligned_data_end - data_begin);
			data_begin = aligned_data_end;
			single_chunk_aggregator.send(std::move(data_chunk));
		}
	}
#endif
#endif

	std::vector<std::string_view> sorted_agg;
	sorted_agg.reserve(aggregates.size());
	aggregates.for_each([&sorted_agg](const std::string& key, entry_type& entry)
		{
			sorted_agg.push_back(key);
			entry.avg = 0.1f * entry.sum / entry.count;
		});

	std::sort(sorted_agg.begin(), sorted_agg.end());

	/*/
	std::ofstream myres("my-res.txt");
	/*/
	std::ostream& myres = std::cout;
	//*/
	myres.precision(1);
	myres << std::fixed;
	myres << "{";
	for (const auto& place_name : sorted_agg) {
		const auto& entry_data = aggregates[place_name];
		myres << place_name << "=" << (entry_data.min / 10.0) << "/" << entry_data.avg << "/" << (entry_data.max / 10.0) << ", ";
	}
	myres << "}";

	//std::string out_str = myres.str();
	//std::cout << out_str;

	return 0;
}


int main(int argc, char* argv[])
{
#ifdef WIN32
	ElapsedTimer total_timer;
#endif
	int rc = my_main(argc, argv);
#ifdef WIN32
	dump_times();
	total_timer.dump("Total");
#endif
	return rc;
}

