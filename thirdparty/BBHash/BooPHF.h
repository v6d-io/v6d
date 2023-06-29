#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include <iostream>

#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <array>
#include <memory>  // for make_shared
#include <unordered_map>
#include <vector>

namespace vineyard {
namespace detail {
namespace boomphf {
class bphf_serde;
}
}
}

namespace boomphf {

inline uint64_t printPt(pthread_t pt) {
  unsigned char* ptc = (unsigned char*) (void*) (&pt);
  uint64_t res = 0;
  for (size_t i = 0; i < sizeof(pt); i++) {
    res += (unsigned) (ptc[i]);
  }
  return res;
}

////////////////////////////////////////////////////////////////
#pragma mark -
#pragma mark utils
////////////////////////////////////////////////////////////////

// iterator from disk file of uint64_t with buffered read,   todo template
template <typename basetype>
class bfile_iterator
    : public std::iterator<std::forward_iterator_tag, basetype> {
 public:
  bfile_iterator() : _is(nullptr), _pos(0), _inbuff(0), _cptread(0) {
    _buffsize = 10000;
    _buffer = (basetype*) malloc(_buffsize * sizeof(basetype));
  }

  bfile_iterator(const bfile_iterator& cr) {
    _buffsize = cr._buffsize;
    _pos = cr._pos;
    _is = cr._is;
    _buffer = (basetype*) malloc(_buffsize * sizeof(basetype));
    memcpy(_buffer, cr._buffer, _buffsize * sizeof(basetype));
    _inbuff = cr._inbuff;
    _cptread = cr._cptread;
    _elem = cr._elem;
  }

  bfile_iterator(FILE* is) : _is(is), _pos(0), _inbuff(0), _cptread(0) {
    // printf("bf it %p\n",_is);
    _buffsize = 10000;
    _buffer = (basetype*) malloc(_buffsize * sizeof(basetype));
    // int reso = fseek(_is,0,SEEK_SET);
    advance();
  }

  ~bfile_iterator() {
    if (_buffer != NULL)
      free(_buffer);
  }

  basetype const& operator*() { return _elem; }

  bfile_iterator& operator++() {
    advance();
    return *this;
  }

  friend bool operator==(bfile_iterator const& lhs, bfile_iterator const& rhs) {
    if (!lhs._is || !rhs._is) {
      if (!lhs._is && !rhs._is) {
        return true;
      } else {
        return false;
      }
    }
    assert(lhs._is == rhs._is);
    return rhs._pos == lhs._pos;
  }

  friend bool operator!=(bfile_iterator const& lhs, bfile_iterator const& rhs) {
    return !(lhs == rhs);
  }

 private:
  void advance() {
    // printf("_cptread %i _inbuff %i \n",_cptread,_inbuff);

    _pos++;

    if (_cptread >= _inbuff) {
      int res = fread(_buffer, sizeof(basetype), _buffsize, _is);

      // printf("read %i new elem last %llu  %p\n",res,_buffer[res-1],_is);
      _inbuff = res;
      _cptread = 0;

      if (res == 0) {
        _is = nullptr;
        _pos = 0;
        return;
      }
    }

    _elem = _buffer[_cptread];
    _cptread++;
  }
  basetype _elem;
  FILE* _is;
  unsigned long _pos;

  basetype* _buffer;  // for buffered read
  int _inbuff, _cptread;
  int _buffsize;
};

template <typename type_elem>
class file_binary {
 public:
  file_binary(const char* filename) {
    _is = fopen(filename, "rb");

    if (!_is) {
      throw std::invalid_argument("Error opening " + std::string(filename));
    }
  }

  ~file_binary() { fclose(_is); }

  bfile_iterator<type_elem> begin() const {
    return bfile_iterator<type_elem>(_is);
  }

  bfile_iterator<type_elem> end() const { return bfile_iterator<type_elem>(); }

  size_t size() const { return 0; }  // todo ?

 private:
  FILE* _is;
};

inline unsigned int popcount_32(unsigned int x) {
  unsigned int m1 = 0x55555555;
  unsigned int m2 = 0x33333333;
  unsigned int m4 = 0x0f0f0f0f;
  unsigned int h01 = 0x01010101;
  x -= (x >> 1) & m1; /* put count of each 2 bits into those 2 bits */
  x = (x & m2) + ((x >> 2) & m2); /* put count of each 4 bits in */
  x = (x + (x >> 4)) &
      m4; /* put count of each 8 bits in partie droite  4bit piece*/
  return (x * h01) >> 24; /* returns left 8 bits of x + (x<<8) + ... */
}

inline unsigned int popcount_64(uint64_t x) {
  unsigned int low = x & 0xffffffff;
  unsigned int high = (x >> 32LL) & 0xffffffff;

  return (popcount_32(low) + popcount_32(high));
}

///// progress bar
class Progress {
 public:
  int timer_mode;
  struct timeval timestamp;
  double heure_debut, heure_actuelle;
  std::string message;

  uint64_t done;
  uint64_t todo;
  int subdiv;  // progress printed every 1/subdiv of total to do
  double partial;
  int _nthreads;
  std::vector<double> partial_threaded;
  std::vector<uint64_t> done_threaded;

  double steps;  // steps = todo/subidv

  void init(uint64_t ntasks, const char* msg, int nthreads = 1) {
    _nthreads = nthreads;
    message = std::string(msg);
    gettimeofday(&timestamp, NULL);
    heure_debut = timestamp.tv_sec + (timestamp.tv_usec / 1000000.0);

    // fprintf(stderr,"| %-*s |\n",98,msg);

    todo = ntasks;
    done = 0;
    partial = 0;

    partial_threaded.resize(_nthreads);
    done_threaded.resize(_nthreads);

    for (int ii = 0; ii < _nthreads; ii++)
      partial_threaded[ii] = 0;
    for (int ii = 0; ii < _nthreads; ii++)
      done_threaded[ii] = 0;
    subdiv = 1000;
    steps = (double) todo / (double) subdiv;

    if (!timer_mode) {
      fprintf(stderr, "[");
      fflush(stderr);
    }
  }

  void finish() {
    set(todo);
    //  if(timer_mode)
    //  	fprintf(stderr,"\n");
    //  else
    //  	fprintf(stderr,"]\n");

    // fflush(stderr);
    todo = 0;
    done = 0;
    partial = 0;
  }
  void finish_threaded()  // called by only one of the threads
  {
    done = 0;
    // double rem = 0;
    for (int ii = 0; ii < _nthreads; ii++)
      done += (done_threaded[ii]);
    for (int ii = 0; ii < _nthreads; ii++)
      partial += (partial_threaded[ii]);

    finish();
  }
  void inc(uint64_t ntasks_done) {
    done += ntasks_done;
    partial += ntasks_done;

    while (partial >= steps) {
      if (timer_mode) {
        gettimeofday(&timestamp, NULL);
        heure_actuelle = timestamp.tv_sec + (timestamp.tv_usec / 1000000.0);
        double elapsed = heure_actuelle - heure_debut;
        double speed = done / elapsed;
        double rem = (todo - done) / speed;
        if (done > todo)
          rem = 0;
        int min_e = (int) (elapsed / 60);
        elapsed -= min_e * 60;
        int min_r = (int) (rem / 60);
        rem -= min_r * 60;

        //  fprintf(stderr,"%c[%s]  %-5.3g%%   elapsed: %3i min %-2.0f sec
        //  remaining: %3i min %-2.0f sec",13, 		message.c_str(),
        //  		100*(double)done/todo,
        //  		min_e,elapsed,min_r,rem);

      } else {
        fprintf(stderr, "-");
        fflush(stderr);
      }
      partial -= steps;
    }
  }

  void inc(uint64_t ntasks_done,
           int tid)  // threads collaborate to this same progress bar
  {
    partial_threaded[tid] += ntasks_done;
    done_threaded[tid] += ntasks_done;
    while (partial_threaded[tid] >= steps) {
      if (timer_mode) {
        struct timeval timet;
        double now;
        gettimeofday(&timet, NULL);
        now = timet.tv_sec + (timet.tv_usec / 1000000.0);
        uint64_t total_done = 0;
        for (int ii = 0; ii < _nthreads; ii++)
          total_done += (done_threaded[ii]);
        double elapsed = now - heure_debut;
        double speed = total_done / elapsed;
        double rem = (todo - total_done) / speed;
        if (total_done > todo)
          rem = 0;
        int min_e = (int) (elapsed / 60);
        elapsed -= min_e * 60;
        int min_r = (int) (rem / 60);
        rem -= min_r * 60;

        //  fprintf(stderr,"%c[%s]  %-5.3g%%   elapsed: %3i min %-2.0f sec
        //  remaining: %3i min %-2.0f sec",13, 		message.c_str(),
        //  		100*(double)total_done/todo,
        //  		min_e,elapsed,min_r,rem);
      } else {
        fprintf(stderr, "-");
        fflush(stderr);
      }
      partial_threaded[tid] -= steps;
    }
  }

  void set(uint64_t ntasks_done) {
    if (ntasks_done > done)
      inc(ntasks_done - done);
  }
  Progress() : timer_mode(0) {}
  // include timer, to print ETA ?
};

////////////////////////////////////////////////////////////////
#pragma mark -
#pragma mark hasher
////////////////////////////////////////////////////////////////

typedef std::array<uint64_t, 10> hash_set_t;
typedef std::array<uint64_t, 2> hash_pair_t;

template <typename Item>
class HashFunctors {
 public:
  /** Constructor.
   * \param[in] nbFct : number of hash functions to be used
   * \param[in] seed : some initialization code for defining the hash functions.
   */
  HashFunctors() {
    _nbFct = 7;  // use 7 hash func
    _user_seed = 0;
    generate_hash_seed();
  }

  // return one hash
  uint64_t operator()(const Item& key, size_t idx) const {
    return hash64(key, _seed_tab[idx]);
  }

  uint64_t hashWithSeed(const Item& key, uint64_t seed) const {
    return hash64(key, seed);
  }

  // this one returns all the 7 hashes
  // maybe use xorshift instead, for faster hash compute
  hash_set_t operator()(const Item& key) {
    hash_set_t hset;

    for (size_t ii = 0; ii < 10; ii++) {
      hset[ii] = hash64(key, _seed_tab[ii]);
    }
    return hset;
  }

 private:
  inline static uint64_t hash64(Item key, uint64_t seed) {
    uint64_t hash = seed;
    hash ^= (hash << 7) ^ key * (hash >> 3) ^
            (~((hash << 11) + (key ^ (hash >> 5))));
    hash = (~hash) + (hash << 21);
    hash = hash ^ (hash >> 24);
    hash = (hash + (hash << 3)) + (hash << 8);
    hash = hash ^ (hash >> 14);
    hash = (hash + (hash << 2)) + (hash << 4);
    hash = hash ^ (hash >> 28);
    hash = hash + (hash << 31);

    return hash;
  }

  /* */
  void generate_hash_seed() {
    static const uint64_t rbase[MAXNBFUNC] = {
        0xAAAAAAAA55555555ULL, 0x33333333CCCCCCCCULL, 0x6666666699999999ULL,
        0xB5B5B5B54B4B4B4BULL, 0xAA55AA5555335533ULL, 0x33CC33CCCC66CC66ULL,
        0x6699669999B599B5ULL, 0xB54BB54B4BAA4BAAULL, 0xAA33AA3355CC55CCULL,
        0x33663366CC99CC99ULL};

    for (size_t i = 0; i < MAXNBFUNC; ++i) {
      _seed_tab[i] = rbase[i];
    }
    for (size_t i = 0; i < MAXNBFUNC; ++i) {
      _seed_tab[i] = _seed_tab[i] * _seed_tab[(i + 3) % MAXNBFUNC] + _user_seed;
    }
  }

  size_t _nbFct;

  static const size_t MAXNBFUNC = 10;
  uint64_t _seed_tab[MAXNBFUNC];
  uint64_t _user_seed;
};

/* alternative hash functor based on xorshift, taking a single hash functor as
input. we need this 2-functors scheme because HashFunctors won't work with
unordered_map. (rayan)
*/

// wrapper around HashFunctors to return only one value instead of 7
template <typename Item>
class SingleHashFunctor {
 public:
  uint64_t operator()(const Item& key,
                      uint64_t seed = 0xAAAAAAAA55555555ULL) const {
    return hashFunctors.hashWithSeed(key, seed);
  }

 private:
  HashFunctors<Item> hashFunctors;
};

template <typename Item, class SingleHasher_t>
class XorshiftHashFunctors {
  /*  Xorshift128*
      Written in 2014 by Sebastiano Vigna (vigna@acm.org)

      To the extent possible under law, the author has dedicated all copyright
      and related and neighboring rights to this software to the public domain
      worldwide. This software is distributed without any warranty.

      See <http://creativecommons.org/publicdomain/zero/1.0/>. */
  /* This is the fastest generator passing BigCrush without
     systematic failures, but due to the relatively short period it is
     acceptable only for applications with a mild amount of parallelism;
     otherwise, use a xorshift1024* generator.

     The state must be seeded so that it is not everywhere zero. If you have
     a nonzero 64-bit seed, we suggest to pass it twice through
     MurmurHash3's avalanching function. */

  //  uint64_t s[ 2 ];

  uint64_t next(uint64_t* s) {
    uint64_t s1 = s[0];
    const uint64_t s0 = s[1];
    s[0] = s0;
    s1 ^= s1 << 23;                                            // a
    return (s[1] = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0;  // b, c
  }

 public:
  uint64_t h0(hash_pair_t& s, const Item& key) {
    s[0] = singleHasher(key, 0xAAAAAAAA55555555ULL);
    return s[0];
  }

  uint64_t h1(hash_pair_t& s, const Item& key) {
    s[1] = singleHasher(key, 0x33333333CCCCCCCCULL);
    return s[1];
  }

  // return next hash an update state s
  uint64_t next(hash_pair_t& s) {
    uint64_t s1 = s[0];
    const uint64_t s0 = s[1];
    s[0] = s0;
    s1 ^= s1 << 23;                                            // a
    return (s[1] = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0;  // b, c
  }

  // this one returns all the  hashes
  hash_set_t operator()(const Item& key) {
    uint64_t s[2];

    hash_set_t hset;

    hset[0] = singleHasher(key, 0xAAAAAAAA55555555ULL);
    hset[1] = singleHasher(key, 0x33333333CCCCCCCCULL);

    s[0] = hset[0];
    s[1] = hset[1];

    for (size_t ii = 2;
         ii < 10 /* it's much better have a constant here, for inlining; this
                    loop is super performance critical*/
         ;
         ii++) {
      hset[ii] = next(s);
    }

    return hset;
  }

 private:
  SingleHasher_t singleHasher;
};

////////////////////////////////////////////////////////////////
#pragma mark -
#pragma mark iterators
////////////////////////////////////////////////////////////////

template <typename Iterator>
struct iter_range {
  iter_range(Iterator b, Iterator e) : m_begin(b), m_end(e) {}

  Iterator begin() const { return m_begin; }

  Iterator end() const { return m_end; }

  Iterator m_begin, m_end;
};

template <typename Iterator>
iter_range<Iterator> range(Iterator begin, Iterator end) {
  return iter_range<Iterator>(begin, end);
}

////////////////////////////////////////////////////////////////
#pragma mark -
#pragma mark BitVector
////////////////////////////////////////////////////////////////

class bitVector {
 public:
  bitVector() : _size(0) { _bitArray = nullptr; }

  bitVector(uint64_t n) : _size(n) {
    _nchar = (1ULL + n / 64ULL);
    _bitArray = (uint64_t*) calloc(_nchar, sizeof(uint64_t));
  }

  ~bitVector() {
    if (_bitArray != nullptr)
      free(_bitArray);
  }

  // copy constructor
  bitVector(bitVector const& r) {
    _size = r._size;
    _nchar = r._nchar;
    _ranks = r._ranks;
    _bitArray = (uint64_t*) calloc(_nchar, sizeof(uint64_t));
    memcpy(_bitArray, r._bitArray, _nchar * sizeof(uint64_t));
  }

  // Copy assignment operator
  bitVector& operator=(bitVector const& r) {
    if (&r != this) {
      _size = r._size;
      _nchar = r._nchar;
      _ranks = r._ranks;
      if (_bitArray != nullptr)
        free(_bitArray);
      _bitArray = (uint64_t*) calloc(_nchar, sizeof(uint64_t));
      memcpy(_bitArray, r._bitArray, _nchar * sizeof(uint64_t));
    }
    return *this;
  }

  // Move assignment operator
  bitVector& operator=(bitVector&& r) {
    // printf("bitVector move assignment \n");
    if (&r != this) {
      if (_bitArray != nullptr)
        free(_bitArray);

      _size = std::move(r._size);
      _nchar = std::move(r._nchar);
      _ranks = std::move(r._ranks);
      _bitArray = r._bitArray;
      r._bitArray = nullptr;
    }
    return *this;
  }
  // Move constructor
  bitVector(bitVector&& r) : _bitArray(nullptr), _size(0) {
    *this = std::move(r);
  }

  void resize(uint64_t newsize) {
    // printf("bitvector resize from  %llu bits to %llu \n",_size,newsize);
    _nchar = (1ULL + newsize / 64ULL);
    _bitArray = (uint64_t*) realloc(_bitArray, _nchar * sizeof(uint64_t));
    _size = newsize;
  }

  size_t size() const { return _size; }

  uint64_t bitSize() const {
    return (_nchar * 64ULL + _ranks.capacity() * 64ULL);
  }

  // clear whole array
  void clear() { memset(_bitArray, 0, _nchar * sizeof(uint64_t)); }

  // clear collisions in interval, only works with start and size multiple of 64
  void clearCollisions(uint64_t start, size_t size, bitVector* cc) {
    assert((start & 63) == 0);
    assert((size & 63) == 0);
    uint64_t ids = (start / 64ULL);
    for (uint64_t ii = 0; ii < (size / 64ULL); ii++) {
      _bitArray[ids + ii] = _bitArray[ids + ii] & (~(cc->get64(ii)));
    }

    cc->clear();
  }

  // clear interval, only works with start and size multiple of 64
  void clear(uint64_t start, size_t size) {
    assert((start & 63) == 0);
    assert((size & 63) == 0);
    memset(_bitArray + (start / 64ULL), 0, (size / 64ULL) * sizeof(uint64_t));
  }

  // for debug purposes
  void print() const {
    printf("bit array of size %lli: \n", (long long int) _size);
    for (uint64_t ii = 0; ii < _size; ii++) {
      if (ii % 10 == 0)
        printf(" (%llu) ", (long long unsigned int) ii);
      int val = (_bitArray[ii >> 6] >> (ii & 63)) & 1;
      printf("%i", val);
    }
    printf("\n");

    printf("rank array : size %lu \n", _ranks.size());
    for (uint64_t ii = 0; ii < _ranks.size(); ii++) {
      printf("%llu :  %lli,  ", (long long unsigned int) ii,
             (long long int) _ranks[ii]);
    }
    printf("\n");
  }

  // return value at pos
  uint64_t operator[](uint64_t pos) const {
    // unsigned char * _bitArray8 = (unsigned char *) _bitArray;
    // return (_bitArray8[pos >> 3ULL] >> (pos & 7 ) ) & 1;

    return (_bitArray[pos >> 6ULL] >> (pos & 63)) & 1;
  }

  // atomically   return old val and set to 1
  uint64_t atomic_test_and_set(uint64_t pos) {
    uint64_t oldval = __sync_fetch_and_or(_bitArray + (pos >> 6),
                                          (uint64_t)(1ULL << (pos & 63)));

    return (oldval >> (pos & 63)) & 1;
  }

  uint64_t get(uint64_t pos) const { return (*this)[pos]; }

  uint64_t get64(uint64_t cell64) const { return _bitArray[cell64]; }

  // set bit pos to 1
  void set(uint64_t pos) {
    assert(pos < _size);
    //_bitArray [pos >> 6] |=   (1ULL << (pos & 63) ) ;
    __sync_fetch_and_or(_bitArray + (pos >> 6ULL), (1ULL << (pos & 63)));
  }

  // set bit pos to 0
  void reset(uint64_t pos) {
    //_bitArray [pos >> 6] &=   ~(1ULL << (pos & 63) ) ;
    __sync_fetch_and_and(_bitArray + (pos >> 6ULL), ~(1ULL << (pos & 63)));
  }

  // return value of  last rank
  // add offset to  all ranks  computed
  uint64_t build_ranks(uint64_t offset = 0) {
    _ranks.reserve(2 + _size / _nb_bits_per_rank_sample);

    uint64_t curent_rank = offset;
    for (size_t ii = 0; ii < _nchar; ii++) {
      if (((ii * 64) % _nb_bits_per_rank_sample) == 0) {
        _ranks.push_back(curent_rank);
      }
      curent_rank += popcount_64(_bitArray[ii]);
    }

    return curent_rank;
  }

  uint64_t rank(uint64_t pos) const {
    uint64_t word_idx = pos / 64ULL;
    uint64_t word_offset = pos % 64;
    uint64_t block = pos / _nb_bits_per_rank_sample;
    uint64_t r = _ranks[block];
    for (uint64_t w = block * _nb_bits_per_rank_sample / 64; w < word_idx;
         ++w) {
      r += popcount_64(_bitArray[w]);
    }
    uint64_t mask = (uint64_t(1) << word_offset) - 1;
    r += popcount_64(_bitArray[word_idx] & mask);

    return r;
  }

  void save(std::ostream& os) const {
    os.write(reinterpret_cast<char const*>(&_size), sizeof(_size));
    os.write(reinterpret_cast<char const*>(&_nchar), sizeof(_nchar));
    os.write(reinterpret_cast<char const*>(_bitArray),
             (std::streamsize)(sizeof(uint64_t) * _nchar));
    size_t sizer = _ranks.size();
    os.write(reinterpret_cast<char const*>(&sizer), sizeof(size_t));
    os.write(reinterpret_cast<char const*>(_ranks.data()),
             (std::streamsize)(sizeof(_ranks[0]) * _ranks.size()));
  }

  void load(std::istream& is) {
    is.read(reinterpret_cast<char*>(&_size), sizeof(_size));
    is.read(reinterpret_cast<char*>(&_nchar), sizeof(_nchar));
    this->resize(_size);
    is.read(reinterpret_cast<char*>(_bitArray),
            (std::streamsize)(sizeof(uint64_t) * _nchar));

    size_t sizer;
    is.read(reinterpret_cast<char*>(&sizer), sizeof(size_t));
    _ranks.resize(sizer);
    is.read(reinterpret_cast<char*>(_ranks.data()),
            (std::streamsize)(sizeof(_ranks[0]) * _ranks.size()));
  }

 protected:
  friend class vineyard::detail::boomphf::bphf_serde;

  uint64_t* _bitArray;
  // uint64_t* _bitArray;
  uint64_t _size;
  uint64_t _nchar;

  // epsilon =  64 / _nb_bits_per_rank_sample   bits
  // additional size for rank is epsilon * _size
  static const uint64_t _nb_bits_per_rank_sample = 512;  // 512 seems ok
  std::vector<uint64_t> _ranks;
};

////////////////////////////////////////////////////////////////
#pragma mark -
#pragma mark level
////////////////////////////////////////////////////////////////

static inline uint64_t fastrange64(uint64_t word, uint64_t p) {
  // return word %  p;

  return (uint64_t)(((__uint128_t) word * (__uint128_t) p) >> 64);
}

class level {
 public:
  level() {}

  ~level() {}

  uint64_t get(uint64_t hash_raw) {
    //	uint64_t hashi =    hash_raw %  hash_domain; //
    // uint64_t hashi = (uint64_t)(  ((__uint128_t) hash_raw * (__uint128_t)
    // hash_domain) >> 64ULL);
    uint64_t hashi = fastrange64(hash_raw, hash_domain);
    return bitset.get(hashi);
  }

  uint64_t idx_begin;
  uint64_t hash_domain;
  bitVector bitset;
};

////////////////////////////////////////////////////////////////
#pragma mark -
#pragma mark mphf
////////////////////////////////////////////////////////////////

#define NBBUFF 10000
//#define NBBUFF 2

template <typename Range, typename Iterator>
struct thread_args {
  void* boophf;
  Range const* range;
  std::shared_ptr<void>
      it_p; /* used to be "Iterator it" but because of fastmode, iterator is
               polymorphic; TODO: think about whether it should be a unique_ptr
               actually */
  std::shared_ptr<void> until_p; /* to cache the "until" variable */
  int level;
};

// forward declaration

template <typename elem_t, typename Hasher_t, typename Range, typename it_type>
void* thread_processLevel(void* args);

/* Hasher_t returns a single hash when operator()(elem_t key) is called.
   if used with XorshiftHashFunctors, it must have the following operator:
   operator()(elem_t key, uint64_t seed) */
template <typename elem_t, typename Hasher_t>
class mphf {
  /* this mechanisms gets P hashes out of Hasher_t */
  typedef XorshiftHashFunctors<elem_t, Hasher_t> MultiHasher_t;
  // typedef HashFunctors<elem_t> MultiHasher_t; // original code (but only
  // works for int64 keys)  (seems to be as fast as the current xorshift)
  // typedef IndepHashFunctors<elem_t,Hasher_t> MultiHasher_t; //faster than
  // xorshift

 public:
  mphf() : _built(false) {}

  ~mphf() {}

  // allow perc_elem_loaded  elements to be loaded in ram for faster
  // construction (default 3%), set to 0 to desactivate
  template <typename Range>
  mphf(size_t n, Range const& input_range, int num_thread = 1,
       double gamma = 2.0, bool writeEach = true, bool progress = true,
       float perc_elem_loaded = 0.03)
      : _gamma(gamma),
        _hash_domain(size_t(ceil(double(n) * gamma))),
        _nelem(n),
        _num_thread(num_thread),
        _percent_elem_loaded_for_fastMode(perc_elem_loaded),
        _withprogress(progress) {
    if (n == 0)
      return;

    _fastmode = false;

    if (_percent_elem_loaded_for_fastMode > 0.0)
      _fastmode = true;

    if (writeEach) {
      _writeEachLevel = true;
      _fastmode = false;
    } else {
      _writeEachLevel = false;
    }

    setup();

    if (_withprogress) {
      _progressBar.timer_mode = 1;

      // double total_raw = _nb_levels;

      double sum_geom_read = (1.0 / (1.0 - _proba_collision));
      double total_writeEach = sum_geom_read + 1.0;

      double total_fastmode_ram =
          (_fastModeLevel + 1) + (pow(_proba_collision, _fastModeLevel)) *
                                     (_nb_levels - (_fastModeLevel + 1));

      // printf("for info, total work write each  : %.3f    total work inram
      // from level %i : %.3f  total work raw : %.3f
      // \n",total_writeEach,_fastModeLevel,total_fastmode_ram,total_raw);

      if (writeEach) {
        _progressBar.init(_nelem * total_writeEach, "Building BooPHF",
                          num_thread);
      } else if (_fastmode)
        _progressBar.init(_nelem * total_fastmode_ram, "Building BooPHF",
                          num_thread);
      else
        _progressBar.init(_nelem * _nb_levels, "Building BooPHF", num_thread);
    }

    uint64_t offset = 0;
    for (int ii = 0; ii < _nb_levels; ii++) {
      _tempBitset = new bitVector(
          _levels[ii].hash_domain);  // temp collision bitarray for this level

      processLevel(input_range, ii);

      _levels[ii].bitset.clearCollisions(0, _levels[ii].hash_domain,
                                         _tempBitset);

      offset = _levels[ii].bitset.build_ranks(offset);

      delete _tempBitset;
    }

    if (_withprogress)
      _progressBar.finish_threaded();

    _lastbitsetrank = offset;

    // printf("used temp ram for construction : %lli MB
    // \n",setLevelFastmode.capacity()* sizeof(elem_t) /1024ULL/1024ULL);

    std::vector<elem_t>().swap(
        setLevelFastmode);  // clear setLevelFastmode reallocating

    pthread_mutex_destroy(&_mutex);

    _built = true;
  }

  uint64_t lookup(const elem_t& elem) {
    if (!_built)
      return ULLONG_MAX;

    // auto hashes = _hasher(elem);
    uint64_t non_minimal_hp, minimal_hp;

    hash_pair_t bbhash;
    bbhash[0] = 0;
    bbhash[1] = 0;
    int level;
    uint64_t level_hash = getLevel(bbhash, elem, &level);

    if (level == (_nb_levels - 1)) {
      auto in_final_map = _final_hash.find(elem);
      if (in_final_map == _final_hash.end()) {
        // elem was not in orignal set of keys
        return ULLONG_MAX;  //  means elem not in set
      } else {
        minimal_hp = in_final_map->second + _lastbitsetrank;
        // printf("lookup %llu  level %i   --> %llu \n",elem,level,minimal_hp);

        return minimal_hp;
      }
      //				minimal_hp = _final_hash[elem] +
      //_lastbitsetrank; 				return minimal_hp;
    } else {
      // non_minimal_hp =  level_hash %  _levels[level].hash_domain; // in fact
      // non minimal hp would be  + _levels[level]->idx_begin
      non_minimal_hp = fastrange64(level_hash, _levels[level].hash_domain);
    }
    minimal_hp = _levels[level].bitset.rank(non_minimal_hp);
    //	printf("lookup %llu  level %i   --> %llu \n",elem,level,minimal_hp);

    return minimal_hp;
  }

  uint64_t nbKeys() const { return _nelem; }

  uint64_t totalBitSize() {
    uint64_t totalsizeBitset = 0;
    for (int ii = 0; ii < _nb_levels; ii++) {
      totalsizeBitset += _levels[ii].bitset.bitSize();
    }

    uint64_t totalsize =
        totalsizeBitset +
        _final_hash.size() * 42 *
            8;  // unordered map takes approx 42B per elem [personal test] (42B
                // with uint64_t key, would be larger for other type of elem)

    printf("Bitarray    %" PRIu64 "  bits (%.2f %%)   (array + ranks )\n",
           totalsizeBitset, 100 * (float) totalsizeBitset / totalsize);
    printf(
        "Last level hash  %12lu  bits (%.2f %%) (nb in last level hash %lu)\n",
        _final_hash.size() * 42 * 8,
        100 * (float) (_final_hash.size() * 42 * 8) / totalsize,
        _final_hash.size());
    return totalsize;
  }

  template <typename Iterator>  // typename Range,
  void pthread_processLevel(std::vector<elem_t>& buffer,
                            std::shared_ptr<Iterator> shared_it,
                            std::shared_ptr<Iterator> until_p, int i) {
    uint64_t nb_done = 0;
    int tid = __sync_fetch_and_add(&_nb_living, 1);
    auto until = *until_p;
    uint64_t inbuff = 0;

    uint64_t writebuff = 0;
    std::vector<elem_t>& myWriteBuff = bufferperThread[tid];

    for (bool isRunning = true; isRunning;) {
      // safely copy n items into buffer
      pthread_mutex_lock(&_mutex);
      for (; inbuff < NBBUFF && (*shared_it) != until; ++(*shared_it)) {
        buffer[inbuff] = *(*shared_it);
        inbuff++;
      }
      if ((*shared_it) == until)
        isRunning = false;
      pthread_mutex_unlock(&_mutex);

      // do work on the n elems of the buffer
      //	printf("filling input  buff \n");

      for (uint64_t ii = 0; ii < inbuff; ii++) {
        elem_t val = buffer[ii];
        // printf("processing %llu  level %i\n",val, i);

        // auto hashes = _hasher(val);
        hash_pair_t bbhash;
        bbhash[0] = 0;
        bbhash[1] = 0;
        int level;
        uint64_t level_hash;
        if (_writeEachLevel)
          getLevel(bbhash, val, &level, i, i - 1);
        else
          getLevel(bbhash, val, &level, i);

        // uint64_t level_hash = getLevel(bbhash,val,&level, i);

        //__sync_fetch_and_add(& _cptTotalProcessed,1);

        if (level == i)  // insert into lvl i
        {
          //	__sync_fetch_and_add(& _cptLevel,1);

          if (_fastmode && i == _fastModeLevel) {
            uint64_t idxl2 =
                __sync_fetch_and_add(&_idxLevelsetLevelFastmode, 1);
            // si depasse taille attendue pour setLevelFastmode, fall back sur
            // slow mode mais devrait pas arriver si hash ok et proba avec nous
            if (idxl2 >= setLevelFastmode.size())
              _fastmode = false;
            else
              setLevelFastmode[idxl2] = val;  // create set for fast mode
          }

          // insert to level i+1 : either next level of the cascade or final
          // hash if last level reached
          if (i == _nb_levels - 1)  // stop cascade here, insert into exact hash
          {
            uint64_t hashidx = __sync_fetch_and_add(&_hashidx, 1);

            pthread_mutex_lock(&_mutex);  // see later if possible to avoid
                                          // this, mais pas bcp item vont la
            // calc rank de fin  precedent level qq part, puis init hashidx avec
            // ce rank, direct minimal, pas besoin inser ds bitset et rank
            _final_hash[val] = hashidx;
            pthread_mutex_unlock(&_mutex);
          } else {
            // ils ont reach ce level
            // insert elem into curr level on disk --> sera utilise au level+1 ,
            // (mais encore besoin filtre)

            if (_writeEachLevel && i > 0 && i < _nb_levels - 1) {
              if (writebuff >= NBBUFF) {
                // flush buffer
                flockfile(_currlevelFile);
                fwrite(myWriteBuff.data(), sizeof(elem_t), writebuff,
                       _currlevelFile);
                funlockfile(_currlevelFile);
                writebuff = 0;
              }

              myWriteBuff[writebuff++] = val;
            }

            // computes next hash

            if (level == 0)
              level_hash = _hasher.h0(bbhash, val);
            else if (level == 1)
              level_hash = _hasher.h1(bbhash, val);
            else {
              level_hash = _hasher.next(bbhash);
            }
            insertIntoLevel(level_hash, i);  // should be safe
          }
        }

        nb_done++;
        if ((nb_done & 1023) == 0 && _withprogress) {
          _progressBar.inc(nb_done, tid);
          nb_done = 0;
        }
      }

      inbuff = 0;
    }

    if (_writeEachLevel && writebuff > 0) {
      // flush buffer
      flockfile(_currlevelFile);
      fwrite(myWriteBuff.data(), sizeof(elem_t), writebuff, _currlevelFile);
      funlockfile(_currlevelFile);
      writebuff = 0;
    }
  }

  void save(std::ostream& os) const {
    os.write(reinterpret_cast<char const*>(&_gamma), sizeof(_gamma));
    os.write(reinterpret_cast<char const*>(&_nb_levels), sizeof(_nb_levels));
    os.write(reinterpret_cast<char const*>(&_lastbitsetrank),
             sizeof(_lastbitsetrank));
    os.write(reinterpret_cast<char const*>(&_nelem), sizeof(_nelem));
    for (int ii = 0; ii < _nb_levels; ii++) {
      _levels[ii].bitset.save(os);
    }

    // save final hash
    size_t final_hash_size = _final_hash.size();

    os.write(reinterpret_cast<char const*>(&final_hash_size), sizeof(size_t));

    // typename std::unordered_map<elem_t,uint64_t,Hasher_t>::iterator
    for (auto it = _final_hash.begin(); it != _final_hash.end(); ++it) {
      os.write(reinterpret_cast<char const*>(&(it->first)), sizeof(elem_t));
      os.write(reinterpret_cast<char const*>(&(it->second)), sizeof(uint64_t));
    }
  }

  void load(std::istream& is) {
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(_gamma));
    is.read(reinterpret_cast<char*>(&_nb_levels), sizeof(_nb_levels));
    is.read(reinterpret_cast<char*>(&_lastbitsetrank), sizeof(_lastbitsetrank));
    is.read(reinterpret_cast<char*>(&_nelem), sizeof(_nelem));

    _levels.resize(_nb_levels);

    for (int ii = 0; ii < _nb_levels; ii++) {
      //_levels[ii].bitset = new bitVector();
      _levels[ii].bitset.load(is);
    }

    // mini setup, recompute size of each level
    _proba_collision =
        1.0 - pow(((_gamma * (double) _nelem - 1) / (_gamma * (double) _nelem)),
                  _nelem - 1);
    uint64_t previous_idx = 0;
    _hash_domain = (size_t)(ceil(double(_nelem) * _gamma));
    for (int ii = 0; ii < _nb_levels; ii++) {
      //_levels[ii] = new level();
      _levels[ii].idx_begin = previous_idx;
      _levels[ii].hash_domain =
          (((uint64_t)(_hash_domain * pow(_proba_collision, ii)) + 63) / 64) *
          64;
      if (_levels[ii].hash_domain == 0)
        _levels[ii].hash_domain = 64;
      previous_idx += _levels[ii].hash_domain;
    }

    // restore final hash

    _final_hash.clear();
    size_t final_hash_size;

    is.read(reinterpret_cast<char*>(&final_hash_size), sizeof(size_t));

    for (unsigned int ii = 0; ii < final_hash_size; ii++) {
      elem_t key;
      uint64_t value;

      is.read(reinterpret_cast<char*>(&key), sizeof(elem_t));
      is.read(reinterpret_cast<char*>(&value), sizeof(uint64_t));

      _final_hash[key] = value;
    }
    _built = true;
  }

 private:
  void setup() {
    pthread_mutex_init(&_mutex, NULL);

    _pid = getpid() + printPt(pthread_self());  // + pthread_self();
    // printf("pt self %llu  pid %i \n",printPt(pthread_self()),_pid);

    _cptTotalProcessed = 0;

    if (_fastmode) {
      setLevelFastmode.resize(_percent_elem_loaded_for_fastMode *
                              (double) _nelem);
    }

    bufferperThread.resize(_num_thread);
    if (_writeEachLevel) {
      for (int ii = 0; ii < _num_thread; ii++) {
        bufferperThread[ii].resize(NBBUFF);
      }
    }

    _proba_collision =
        1.0 - pow(((_gamma * (double) _nelem - 1) / (_gamma * (double) _nelem)),
                  _nelem - 1);

    // double sum_geom =_gamma * ( 1.0 +  _proba_collision / (1.0 -
    // _proba_collision));
    // printf("proba collision %f  sum_geom  %f \n",_proba_collision,sum_geom);

    _nb_levels = 25;
    _levels.resize(_nb_levels);

    // build levels
    uint64_t previous_idx = 0;
    for (int ii = 0; ii < _nb_levels; ii++) {
      _levels[ii].idx_begin = previous_idx;

      // round size to nearest superior multiple of 64, makes it easier to clear
      // a level
      _levels[ii].hash_domain =
          (((uint64_t)(_hash_domain * pow(_proba_collision, ii)) + 63) / 64) *
          64;
      if (_levels[ii].hash_domain == 0)
        _levels[ii].hash_domain = 64;
      previous_idx += _levels[ii].hash_domain;

      // printf("build level %i bit array : start %12llu, size %12llu
      // ",ii,_levels[ii]->idx_begin,_levels[ii]->hash_domain ); printf("
      // expected elems : %.2f %% total \n",100.0*pow(_proba_collision,ii));
    }

    for (int ii = 0; ii < _nb_levels; ii++) {
      if (pow(_proba_collision, ii) < _percent_elem_loaded_for_fastMode) {
        _fastModeLevel = ii;
        // printf("fast mode level :  %i \n",ii);
        break;
      }
    }
  }

  // compute level and returns hash of last level reached
  uint64_t getLevel(hash_pair_t& bbhash, const elem_t& val, int* res_level,
                    int maxlevel = 100, int minlevel = 0)
  // uint64_t getLevel(hash_pair_t & bbhash, elem_t val,int * res_level, int
  // maxlevel = 100, int minlevel =0)

  {
    int level = 0;
    uint64_t hash_raw = 0;

    for (int ii = 0; ii < (_nb_levels - 1) && ii < maxlevel; ii++) {
      // calc le hash suivant
      if (ii == 0)
        hash_raw = _hasher.h0(bbhash, val);
      else if (ii == 1)
        hash_raw = _hasher.h1(bbhash, val);
      else {
        hash_raw = _hasher.next(bbhash);
      }

      if (ii >= minlevel && _levels[ii].get(hash_raw))  //
      // if(  _levels[ii].get(hash_raw) ) //

      {
        break;
      }

      level++;
    }

    *res_level = level;
    return hash_raw;
  }

  // insert into bitarray
  void insertIntoLevel(uint64_t level_hash, int i) {
    //	uint64_t hashl =  level_hash % _levels[i].hash_domain;
    uint64_t hashl = fastrange64(level_hash, _levels[i].hash_domain);

    if (_levels[i].bitset.atomic_test_and_set(hashl)) {
      _tempBitset->atomic_test_and_set(hashl);
    }
  }

  // loop to insert into level i
  template <typename Range>
  void processLevel(Range const& input_range, int i) {
    ////alloc the bitset for this level
    _levels[i].bitset = bitVector(_levels[i].hash_domain);
    ;

    // printf("---process level %i   wr %i fast %i
    // ---\n",i,_writeEachLevel,_fastmode);

    char fname_old[1000];
    sprintf(fname_old, "temp_p%i_level_%i", _pid, i - 2);

    char fname_curr[1000];
    sprintf(fname_curr, "temp_p%i_level_%i", _pid, i);

    char fname_prev[1000];
    sprintf(fname_prev, "temp_p%i_level_%i", _pid, i - 1);

    if (_writeEachLevel) {
      // file management :

      if (i > 2)  // delete previous file
      {
        unlink(fname_old);
      }

      if (i < _nb_levels - 1 && i > 0)  // create curr file
      {
        _currlevelFile = fopen(fname_curr, "w");
      }
    }

    _cptLevel = 0;
    _hashidx = 0;
    _idxLevelsetLevelFastmode = 0;
    _nb_living = 0;
    // create  threads
    pthread_t* tab_threads = new pthread_t[_num_thread];
    typedef decltype(input_range.begin()) it_type;
    thread_args<Range, it_type> t_arg;  // meme arg pour tous
    t_arg.boophf = this;
    t_arg.range = &input_range;
    t_arg.it_p = std::static_pointer_cast<void>(
        std::make_shared<it_type>(input_range.begin()));
    t_arg.until_p = std::static_pointer_cast<void>(
        std::make_shared<it_type>(input_range.end()));

    t_arg.level = i;

    if (_writeEachLevel && (i > 1)) {
      auto data_iterator_level = file_binary<elem_t>(fname_prev);

      typedef decltype(data_iterator_level.begin()) disklevel_it_type;

      // data_iterator_level.begin();
      t_arg.it_p = std::static_pointer_cast<void>(
          std::make_shared<disklevel_it_type>(data_iterator_level.begin()));
      t_arg.until_p = std::static_pointer_cast<void>(
          std::make_shared<disklevel_it_type>(data_iterator_level.end()));

      for (int ii = 0; ii < _num_thread; ii++)
        pthread_create(
            &tab_threads[ii], NULL,
            thread_processLevel<elem_t, Hasher_t, Range, disklevel_it_type>,
            &t_arg);  //&t_arg[ii]

      // must join here before the block is closed and file_binary is destroyed
      // (and closes the file)
      for (int ii = 0; ii < _num_thread; ii++) {
        pthread_join(tab_threads[ii], NULL);
      }
    }

    else {
      if (_fastmode && i >= (_fastModeLevel + 1)) {
        /* we'd like to do t_arg.it = data_iterator.begin() but types are
         different;
         so, casting to (void*) because of that; and we remember the type in the
         template */
        typedef decltype(setLevelFastmode.begin()) fastmode_it_type;
        t_arg.it_p = std::static_pointer_cast<void>(
            std::make_shared<fastmode_it_type>(setLevelFastmode.begin()));
        t_arg.until_p = std::static_pointer_cast<void>(
            std::make_shared<fastmode_it_type>(setLevelFastmode.end()));

        /* we'd like to do t_arg.it = data_iterator.begin() but types are
         different;
         so, casting to (void*) because of that; and we remember the type in the
         template */

        for (int ii = 0; ii < _num_thread; ii++)
          pthread_create(
              &tab_threads[ii], NULL,
              thread_processLevel<elem_t, Hasher_t, Range, fastmode_it_type>,
              &t_arg);  //&t_arg[ii]

      } else {
        for (int ii = 0; ii < _num_thread; ii++)
          pthread_create(&tab_threads[ii], NULL,
                         thread_processLevel<elem_t, Hasher_t, Range,
                                             decltype(input_range.begin())>,
                         &t_arg);  //&t_arg[ii]
      }
      // joining
      for (int ii = 0; ii < _num_thread; ii++) {
        pthread_join(tab_threads[ii], NULL);
      }
    }
    // printf("\ngoing to level %i  : %llu elems  %.2f %%  expected : %.2f %%
    // \n",i,_cptLevel,100.0* _cptLevel/(float)_nelem,100.0*
    // pow(_proba_collision,i) );

    // printf("\ncpt total processed %llu \n",_cptTotalProcessed);
    if (_fastmode &&
        i == _fastModeLevel)  // shrink to actual number of elements in set
    {
      // printf("\nresize setLevelFastmode to %lli
      // \n",_idxLevelsetLevelFastmode);
      setLevelFastmode.resize(_idxLevelsetLevelFastmode);
    }
    delete[] tab_threads;

    if (_writeEachLevel) {
      if (i < _nb_levels - 1 && i > 0) {
        fflush(_currlevelFile);
        fclose(_currlevelFile);
      }

      if (i == _nb_levels - 1)  // delete last file
      {
        unlink(fname_prev);
      }
    }
  }

 private:
  friend class vineyard::detail::boomphf::bphf_serde;

  // level ** _levels;
  std::vector<level> _levels;
  int _nb_levels;
  MultiHasher_t _hasher;
  bitVector* _tempBitset;

  double _gamma;
  uint64_t _hash_domain;
  uint64_t _nelem;
  std::unordered_map<elem_t, uint64_t, Hasher_t> _final_hash;
  Progress _progressBar;
  int _nb_living;
  int _num_thread;
  uint64_t _hashidx;
  double _proba_collision;
  uint64_t _lastbitsetrank;
  uint64_t _idxLevelsetLevelFastmode;
  uint64_t _cptLevel;
  uint64_t _cptTotalProcessed;

  // fast build mode , requires  that _percent_elem_loaded_for_fastMode % elems
  // are loaded in ram
  float _percent_elem_loaded_for_fastMode;
  bool _fastmode;
  std::vector<elem_t> setLevelFastmode;
  //	std::vector< elem_t > setLevelFastmode_next; // todo shrinker le set e
  // nram a chaque niveau  ?

  std::vector<std::vector<elem_t>> bufferperThread;

  int _fastModeLevel;
  bool _withprogress;
  bool _built;
  bool _writeEachLevel;
  FILE* _currlevelFile;
  int _pid;

 public:
  pthread_mutex_t _mutex;
};

////////////////////////////////////////////////////////////////
#pragma mark -
#pragma mark threading
////////////////////////////////////////////////////////////////

template <typename elem_t, typename Hasher_t, typename Range, typename it_type>
void* thread_processLevel(void* args) {
  if (args == NULL)
    return NULL;

  thread_args<Range, it_type>* targ = (thread_args<Range, it_type>*) args;

  mphf<elem_t, Hasher_t>* obw = (mphf<elem_t, Hasher_t>*) targ->boophf;
  int level = targ->level;
  std::vector<elem_t> buffer;
  buffer.resize(NBBUFF);

  pthread_mutex_t* mutex = &obw->_mutex;

  pthread_mutex_lock(
      mutex);  // from comment above: "//get starting iterator for this thread,
               // must be protected (must not be currently used by other thread
               // to copy elems in buff)"
  std::shared_ptr<it_type> startit =
      std::static_pointer_cast<it_type>(targ->it_p);
  std::shared_ptr<it_type> until_p =
      std::static_pointer_cast<it_type>(targ->until_p);
  pthread_mutex_unlock(mutex);

  obw->pthread_processLevel(buffer, startit, until_p, level);

  return NULL;
}
}  // namespace boomphf
