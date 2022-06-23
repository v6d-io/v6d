Adding Custom Data Types in C++
===============================

Vineyard has already support a set of efficient builtin data types in
the C++ SDK, e.g., :code:`Vector`, :code:`HashMap`, :code:`Tensor`,
:code:`DataFrame`, :code:`Table` and :code:`Graph`, (see :ref:`cpp-api`).
However there are still scenarios where users need to develop their
own data structures and efficiently share the data with Vineyard. Custom
C++ data types could be easily added by following this step-by-step tutorial.

    Note that this tutorial includes code that could be auto-generated for
    keeping clear about the design internals and helping developers get a whole
    picture about how vineyard client works.

Vineyard Objects
----------------

Vineyard has a base class :code:`vineyard::Objects`, and a corresponding
base class :code:`Vineyard::ObjectBuilder` for builders as follows,

.. code:: cpp

    class Object {
      public:
        static std::unique_ptr<Object> Create() {
            ...
        }

        virtual void Construct(const ObjectMeta& meta);
    }

and the builder

.. code:: cpp

    class ObjectBuilder {
        virtual Status Build(Client& client) override = 0;

        virtual std::shared_ptr<Object> _Seal(Client& client) = 0;
    }

Where the object is the base class for user-defined data types, and the
builders is responsible for placing the data into vineyard.


reg.docker.alibaba-inc.com/v6d/graphscope:2f40c443

Adding Your Own Types
---------------------

We taking defining a custom :code:`Vector` type as example. Basically,
a :code:`Vector` contains a :code:`vineyard::Blob` as payload, and metadata
like :code:`dtype` and :code:`size` as well.

The class for :code:`Vector` usually looks like

.. code:: cpp

    template <typename T>
    class Vector {
    private:
        size_t size;
        const T *data = nullptr;
    public:
        Vector(): size(0), data(nullptr) {
        }

        Vector(const int size, const T *data): size(size), data(data) {
        }

        size_t length() const {
            return size;
        }

        const T& operator[](size_t index) {
            assert(index < size);
            return data[index];
        }
    };

Defining the data structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first migrate the existing :code:`Vector<T>` to vineyard's :code:`Object`,

.. code:: diff

     template <typename T>
    -class Vector {
    +class Vector: public vineyard::Registered<Vector<T>> {
       private:
         size_t size;
         T *data = nullptr;
       public:
    +    static std::unique_ptr<Object> Create() __attribute__((used)) {
    +        return std::static_pointer_cast<Object>(
    +            std::unique_ptr<Vector<T>>{
    +                new Vector<T>()});
    +    }
    +
         Vector(): size(0), data(nullptr) {
         }
     
         Vector(const int size, const T *data): size(size), data(data) {
         }

         ...
     }

Note the two changes above,

+ inherits from :code:`vineyard::Registered<Vector<T>>`

    :code:`vineyard::Registered<T>` is a helper to generate some static
    initialization stubs to register the data type :code:`T` to the type
    resolving factory, and associate the type :code:`T` with its typename.
    The typename is the auto-generated readable name for C++ types, e.g.,
    :code:`"Vector<int32>"` for :code:`Vector<int32_t>`.

+ The zero-parameter static constructor :code:`Create()`

    :code:`Create()` is a static function that will be registered to
    the resolving factory by helper :code:`vineyard::Registered<T>` and
    used to construct a instance of type :code:`T` first when getting objects
    from vineyard.

    Vineyard client looks up the static constructor by :code:`typename` in
    the metadata of vineyard objects store in the daemon server.

To obtain the object :code:`Vector<T>` from vineyard's metadata, we need to
implements a `Construct` method as well. The :code:`Construct` method takes
a :code:`vineyard::ObjectMeta` as input, and retrieve metadata as well as
members from the metadata to fill its own data members. The memory in member
:code:`buffer` (a :code:`vineyard::Blob`) is shared using memory mapping,
without the cost of copying.

.. code:: diff

     template <typename T>
     class Vector: public vineyard::Registered<Vector<T>> {
       public:
         ...
 
    +    void Construct(const ObjectMeta& meta) override {
    +      this->size = meta.GetKeyValue<size_t>("size");
    +
    +      auto buffer = std::dynamic_pointer_cast<Blob>(meta.GetMember("buffer"));
    +      this->data = reinterpret_cast<const T *>(buffer->data());
    +    }
    +
         ...
     }

Implements the builder
^^^^^^^^^^^^^^^^^^^^^^

Next, we go the builder part. The :code:`vineyard::ObjectBuilder` contains two
part,

+ :code:`Build()`: this method is responsible for storing blobs of custom data
  structures into vineyard

+ :code:`_Seal()`: this method is responsible for generate the corresponding
  metadata and putting the metadata into vineyard

For our :code:`Vector<T>` type, we first define a general vector builder,

.. code:: cpp

    template <typename T>
    class VectorBuilder {
      private:
        std::unique_ptr<BlobWriter> buffer_builder;
        std::size_t size;
        T *data;
    
      public:
        VectorBuilder(size_t size): size(size) {
          data = static_cast<T *>(malloc(sizeof(T) * size));
        }
    
        T& operator[](size_t index) {
          assert(index < size);
          return data[index];
        }
    };
    
The builder allocate the required memory based on required :code:`size` to contain
the elements, and a `[]` operator to fill the data in.

Now we adapts the builder above as a `ObjectBuilder` in vineyard,

.. code:: diff

     template <typename T>
    -class VectorBuilder {
    +class VectorBuilder: public vineyard::ObjectBuilder {
       private:
         std::unique_ptr<BlobWriter> buffer_builder;
         std::size_t size;
         T *data;
     
       public:
         VectorBuilder(size_t size): size(size) {
           data = static_cast<T *>(malloc(sizeof(T) * size));
         }
     
    +    Status Build(Client& client) override {
    +      VINEYARD_CHECK_OK(client.CreateBlob(size * sizeof(T), buffer_builder));
    +      memcpy(buffer_builder->data(), data, size * sizeof(T));
    +      return Status::OK();
    +    }
    +
    +    std::shared_ptr<Object> _Seal(Client& client) override {
    +      VINEYARD_CHECK_OK(this->Build(client));
    +
    +      auto vec = std::make_shared<Vector<int>>();
    +      auto buffer = std::dynamic_pointer_cast<vineyard::Blob>(
    +        this->buffer_builder->Seal(client));
    +      vec->size = size;
    +      vec->data = reinterpret_cast<const T *>(buffer->data());
    +
    +      vec->meta_.SetTypeName(vineyard::type_name<Vector<T>>());
    +      vec->meta_.SetNBytes(size * sizeof(T));
    +      vec->meta_.AddKeyValue("size", size);
    +      vec->meta_.AddMember("buffer", buffer);
    +      VINEYARD_CHECK_OK(client.CreateMetaData(vec->meta_, vec->id_));
    +
    +      return vec;
    +    }
    +
         T& operator[](size_t index) {
           assert(index < size);
           return data[index];
         }
     };

Note that the builder needs to directly access the private data member of
:code:`Vector<T>`, thus we need to makes the builder as a friend class of
our vector type,

.. code:: diff

     template <typename T>
     class Vector: public vineyard::Registered<Vector<T>> {
    
         const T& operator[](size_t index) {
           assert(index < size);
           return data[index];
         }
    +  
    +  friend class VectorBuilder<T>;
     };
    
As you can see in the above example, there are many boilerplate snippets
in the builder and constructor. They are be auto-generated from the layout
of class :code:`Vector<T>` based on the static analysis of user's source code.

Now it should work!
^^^^^^^^^^^^^^^^^^^

Finally we are able to build our custom data types into vineyard and retrieve
it back, using vineyard client,

.. code:: cpp

    int main(int argc, char** argv) {
        std::string ipc_socket = std::string(argv[1]);

        Client client;
        VINEYARD_CHECK_OK(client.Connect(ipc_socket));
        LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

        auto builder = VectorBuilder<int>(3);
        builder[0] = 1;
        builder[1] = 2;
        builder[2] = 3;
        auto result = builder.Seal(client);

        auto vec = std::dynamic_pointer_cast<Vector<int>>(client.GetObject(result->id()));
        for (size_t index = 0; index < vec->length(); ++index) {
            std::cout << "element at " << index << " is: " << (*vec)[index] << std::endl;
        }
    }

Builders and Resolvers in other Languages
-----------------------------------------

Vineyard keeps the same design principle for SDKs in other languages, e.g.,
Java and Python. For an example in Python about the vineyard objects and its
builders, see also :ref:`divein-builder-resolver`.

As described in the example above, there are a lots of boilerplate code when
defining the constructor and builder. To make the integration with vineyard
easier, a code generator is already on the way to generate SDKs in different
languages based on a C++-like DSL, just stay tuned!

For a preview about how the code generator works, please refer to `array.vineyard-mod`_
and `arrow.vineyard-mod`_.


.. _array.vineyard-mod: https://github.com/v6d-io/v6d/blob/main/modules/basic/ds/array.vineyard-mod
.. _arrow.vineyard-mod: https://github.com/v6d-io/v6d/blob/main/modules/basic/ds/arrow.vineyard-mod
