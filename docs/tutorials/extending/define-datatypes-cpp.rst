.. _define-cpp-types:

Defining Custom Data Types in C++
=================================

Vineyard provides an extensive set of efficient built-in data types in
its C++ SDK, such as :code:`Vector`, :code:`HashMap`, :code:`Tensor`,
:code:`DataFrame`, :code:`Table`, and :code:`Graph` (refer to :ref:`cpp-api`).
However, there may be situations where users need to develop their
own data structures and share the data efficiently with Vineyard. This
step-by-step tutorial guides you through the process of adding custom
C++ data types with ease.

.. note::

    This tutorial includes code snippets that could be auto-generated to
    provide a clear understanding of the design internals and to help
    developers grasp the overall functionality of the Vineyard client.

``Object`` and ``ObjectBuilder``
--------------------------------

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

Defining Your Custom Type
-------------------------

Let's take the example of defining a custom `Vector` type. Essentially,
a `Vector` consists of a `vineyard::Blob` as its payload, along with
metadata such as `dtype` and `size`.

The class definition for the `Vector` type typically appears as follows:

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

Registering C++ Types
---------------------

First, we need to adapt the existing :code:`Vector<T>` to become a Vineyard
:code:`Object`,

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

Observe the two key modifications above:

+ Inheriting from :code:`vineyard::Registered<Vector<T>>`:

  :code:`vineyard::Registered<T>` serves as a helper to generate static
  initialization stubs, registering the data type :code:`T` with the type
  resolving factory and associating the type :code:`T` with its typename.
  The typename is an auto-generated, human-readable name for C++ types, e.g.,
  :code:`"Vector<int32>"` for :code:`Vector<int32_t>`.

+ Implementing the zero-parameter static constructor :code:`Create()`:

  :code:`Create()` is a static function registered with the
  resolving factory by the helper :code:`vineyard::Registered<T>`. It is
  used to construct an instance of type :code:`T` when retrieving objects
  from Vineyard.

  The Vineyard client locates the static constructor using the :code:`typename`
  found in the metadata of Vineyard objects stored in the daemon server.

To retrieve the object :code:`Vector<T>` from Vineyard's metadata, we need to
implement a `Construct` method as well. The :code:`Construct` method takes
a :code:`vineyard::ObjectMeta` as input and extracts metadata and
members from it to populate its own data members. The memory in the member
:code:`buffer` (a :code:`vineyard::Blob`) is shared using memory mapping,
eliminating the need for copying.

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

Builder
-------

Moving on to the builder section, the :code:`vineyard::ObjectBuilder` consists of two parts:

+ :code:`Build()`: This method is responsible for storing the blobs of custom data
  structures into Vineyard.

+ :code:`_Seal()`: This method is responsible for generating the corresponding metadata
  and inserting the metadata into Vineyard.

For our :code:`Vector<T>` type, let's first define a general vector builder:

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

The builder allocates the necessary memory based on the specified :code:`size` to accommodate
the elements and provides a `[]` operator to populate the data.

Next, we adapt the above builder as a `ObjectBuilder` in Vineyard,

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
    +      RETURN_ON_ERROR(client.CreateBlob(size * sizeof(T), buffer_builder));
    +      memcpy(buffer_builder->data(), data, size * sizeof(T));
    +      return Status::OK();
    +    }
    +
    +    Status _Seal(Client& client, std::shared_ptr<Object> &object) override {
    +      RETURN_ON_ERROR(this->Build(client));
    +
    +      auto vec = std::make_shared<Vector<int>>();
           object = vec;
    +      std::shared_ptr<Object> buffer_object;
    +      RETURN_ON_ERROR(this->buffer_builder->Seal(client, buffer_object));
    +      auto buffer = std::dynamic_pointer_cast<Blob>(buffer_object);
    +      vec->size = size;
    +      vec->data = reinterpret_cast<const T *>(buffer->data());
    +
    +      vec->meta_.SetTypeName(vineyard::type_name<Vector<T>>());
    +      vec->meta_.SetNBytes(size * sizeof(T));
    +      vec->meta_.AddKeyValue("size", size);
    +      vec->meta_.AddMember("buffer", buffer);
    +      return client.CreateMetaData(vec->meta_, vec->id_);
    +    }
    +
         T& operator[](size_t index) {
           assert(index < size);
           return data[index];
         }
     };

To access private member fields and methods, the builder may need to be
added as a friend class of the original type declaration.

.. note::

   Since the builder requires direct access to the private data members of
   :code:`Vector<T>`, it is necessary to declare the builder as a friend class
   of our vector type,

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

In the example above, you may notice that the builder and constructor contain numerous
boilerplate snippets. These can be auto-generated based on the layout of the class
:code:`Vector<T>` through static analysis of the user's source code, streamlining
the process and enhancing readability.

Utilizing Custom Data Types with Vineyard
-----------------------------------------

At this point, we have successfully defined our custom data types and integrated them
with Vineyard. Now, we can demonstrate how to build these custom data types using the
Vineyard client and retrieve them for further processing.

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

Cross-Language Compatibility
----------------------------

Vineyard maintains consistent design principles across SDKs in various languages,
such as Java and Python. For an example of Vineyard objects and their builders in
Python, please refer to :ref:`builder-resolver`.

As demonstrated in the example above, there is a significant amount of boilerplate
code involved in defining constructors and builders. To simplify the integration
with Vineyard, we are developing a code generator that will automatically produce
SDKs in different languages based on a C++-like Domain Specific Language (DSL).
Stay tuned for updates!

For a sneak peek at how the code generator works, please refer to `array.vineyard-mod`_
and `arrow.vineyard-mod`_.

.. _array.vineyard-mod: https://github.com/v6d-io/v6d/blob/main/modules/basic/ds/array.vineyard-mod
.. _arrow.vineyard-mod: https://github.com/v6d-io/v6d/blob/main/modules/basic/ds/arrow.vineyard-mod
