
import random, math, hashlib

class hasher: 
	"""
	Hash table for storing and looking up various contexts associated to an estimated HRF.
	This hashtable uses pen addressing for collisions resolution for lower memory requirements
	for storing Usually hash table used for super quick lookup times as well as insertion and
	deletions (constant time! O(1)) great for applications that require a lot of look ups. Two
	types of Hash Tables below with varied collision resolution function: Chained and open
	addressed. 
	
	Note on Hash Function itself: The hashing function is a bit of an art and design
	with vary across developers. Key notes to remember is that the gold standard for a good hash
	function it will lead too good performance (i.e. spread data out evenly across buckets) and 
	should be easy to both store AND access data. Very easy to develope a bad hash function so be
	sure to look at many examples in many contexts when developing a function for real life use!
	
	Class functions:
		- pyhash() -
		- sha3() - 
    
	Class attributed:

	
	"""
	def __init__(self, context):
		self.min_fill = float(1/3)
		self.max_fill = float(2/3)

		self.size = 0
		self.capacity = 2

		self.collision_count = 0
		self.probe_count = 0

		self.constant = random.uniform(0, 1)

		self.hasher = lambda key : self.sha3(key) # lambda function for quickly switching between hashes for testing intial hash
		self.prober = lambda key, hashkey : self.linear_probe(key, hashkey) # Lambda function for switching between hashed for testing probes

		self.table = [None]*self.capacity # Declare hash table
		self.contexts = [[] for _ in range(self.capacity)]

	def __repr__(self): # Class callback for assessing current capactiy/fill/collision rate
		fill_pct = (self.size / self.capacity) * 100
		collision_pct = (self.collision_count / self.size) * 100 if self.size > 0 else 0.0
		return f"HashTable with {self.capacity} capacity {fill_pct:.1f}% full - {collision_pct:.1f}% Collision Rate"

	# ------------- Hash Table Hash Functions --------------- #
	# Obsolete Hash Functions: MD5, SHA-1, SHA-2 (Not obsolete yet but a matter of time)

	def pyhash(self, key): # Uses a pythons inbuilt hash function which utilitzes SipHash
		# Avoid using this function for times when you have a lot of very close
		# values, especially integers, being inserted into a hash table to avoid primary clustering
		return hash(key) % self.capacity # Utilizes Add-Block-XOR Block Cipher

	def sha3(self, key): # Secure Hash Algorithm 3
		return int.from_bytes(hashlib.sha3_512(self.encode(key)).digest(), 'little') % self.capacity

	def blake2(self, key, subhash = 's'):
		"""
		BLAKE2 hash function, faster than MD5, SHA-1, SHA-
		
		Arguments:
			key (str) - Key to hash
			subhash (str) - 's' for blake2s, 'b'
		"""
		if subhash == 's':
			return int.from_bytes(hashlib.blake2s(self.encode(key)).digest(), 'little') % self.capacity
		if subhash == 'b':
			return int.from_bytes(hashlib.blake2b(self.encode(key)).digest(), 'little') % self.capacity

	def division_hash(self, key):
		"""
		Division hash function
		
		Arguments:
			key (str) - Key to hash

		Returns:
			int - Hash value
		"""
		return sum(self.encode(key)) % self.capacity

	def multiplication_hash(self, key):
		"""
		Multiplication hash function
		
		Arguments:
			key (str) - Key to hash

		Returns:
			int - Hash value
		"""
		return math.floor(self.capacity*((sum(self.encode(key))*self.constant) % 1))

	def encode(self, key):
		"""
		Encode a key into bytes for hashing
		
		Arguments:
			key (str) - Key to encode

		Returns
			bytes - Encoded key
		"""
		return bytes(str(key), 'utf-8')

	# ------------ Hash Table Collision Probes -------------- #

	def linear_probe(self, key, hashkey, a = 5, b = 1): # Probing function used for linear hashing
		"""
		Linear probe function

		Arguments:
			key (str) - Key to hash
			hashkey (int) - Initial hashkey to rehash
			a (int) - Linear coefficient
			b (int) - Constant coefficient

		Returns:
			int - New hashkey
		"""
		self.collision_count += 1
		self.probe_count += 1
		return (a * hashkey + b) % self.capacity

	def quad_probe(self, key, hashkey): # Probing function used for quadratic probing
		"""
		Quadratic probe function

		Arguments:
			key (str) - Key to hash
			hashkey (int) - Initial hashkey to rehash
		
		Returns:
			int - New hashkey
		"""
		self.collision_count += 1
		self.probe_count += 1
		return (hashkey + (self.probe_count**2)) % self.capacity

	def double_probe(self, key, hashkey):
		"""
		Double hashing probe function
		
		Arguments:
			key (str) - Key to hash
			hashkey (int) - Initial hashkey to rehash

		Returns:
			int - New hashkey
		"""
		self.collision_count += 1
		self.probe_count += 1
		hashkey = self.linear_probe(key, hashkey) + self.probe_count * self.quad_probe(key, hashkey)
		return hashkey % self.capacity

	# ------------- Core Hash Table Functions ------------- #

	def add(self, key, pointer):
		"""
		Add a pointer under the given key. Multiple pointers may share a
		key — subsequent adds append to the slot's pointer list. Duplicate
		(key, pointer) pairs are deduplicated by identity.

		Arguments:
			key (str) - Key to add to the hash table
			pointer (any) - Pointer to associate with the key

		Returns:
			None
		"""
		fill = float(self.size/self.capacity)
		if self.min_fill > fill or fill > self.max_fill:
			self.resize()
		hashkey = self.hasher(key)
		while self.table[hashkey] is not None:
			if self.table[hashkey] == key: # Key already present — append if new
				if not any(existing is pointer for existing in self.contexts[hashkey]):
					self.contexts[hashkey].append(pointer)
				self.probe_count = 0
				return
			if self.table[hashkey] == '!tombstone!': # If a tombstone was found
				break # Replace tombstone
			hashkey = self.prober(key, hashkey)

		self.table[hashkey] = key # insert the new key into the found hash
		self.contexts[hashkey] = [pointer] # Start a fresh pointer list for this key

		self.size += 1 # Increment size
		self.probe_count = 0 # Reset

	def search(self, key):
		"""
		Search for a key in the hash table and return the list of pointers
		associated with it.

		Arguments:
			key (str) - Key to search for in the hash table

		Returns:
			list - Pointers associated with the key. Empty list on miss.
			A shallow copy is returned so callers can safely iterate and
			mutate their own copy without affecting the hasher's state.
		"""
		hashkey = self.hasher(key)
		steps = 0
		while self.table[hashkey] is not None:
			if self.table[hashkey] == key:
				self.probe_count = 0
				return list(self.contexts[hashkey])
			hashkey = self.prober(key, hashkey)
			steps += 1
			if steps >= self.capacity:  # Full cycle — key not present
				break
		self.probe_count = 0 # Reset the quadratic multiplier probe for the next call
		return []

	def remove(self, key):
		hashkey = self.hasher(key)
		while self.table[hashkey]:
			if self.table[hashkey] == key:
				self.table[hashkey] = '!tombstone!'
				self.contexts[hashkey] = []
				self.size -= 1
				break
			hashkey = self.prober(key, hashkey)
		if self.min_fill > float(self.size/self.capacity): # If table is bellow minimum fill
			self.resize() # resize
		self.probe_count = 0 # Reset the quadratic multiplier probe for the next call

	def resize(self):
		fill = float(self.size/self.capacity)
		old_capacity = self.capacity
		if self.min_fill > fill:
			self.capacity >>= 1
			print(f"Table below minimum fill, decreasing capacity to {self.capacity}")
		else:
			self.capacity <<= 1
			print(f"Table exceeding maximum fill, increasing capacity to {self.capacity}")
		new_table = [None]*self.capacity
		new_hrf_filenames = [[] for _ in range(self.capacity)]
		for ind in range(old_capacity):
			if self.table[ind] and self.table[ind] != '!tombstone!':
				position = self.hasher(self.table[ind])
				while new_table[position] is not None:
					position = self.prober(self.table[ind], position)
				new_table[position] = self.table[ind]
				new_hrf_filenames[position] = self.contexts[ind]
		self.table = new_table
		self.contexts = new_hrf_filenames

