.. _vcdl:

Code Generation for Boilerplate
===============================

Sharing objects between engines consists of two basic steps, defining the
data structure and defining the protocol for formulating the data type
as :ref:`vineyard-objects`. Vineyard provides an auto-generation mechanism
to reduce the burden when integrating with custom data types. The mechanism,
namely **VCDL**, is based on custom annotation :code:`[[shared]]` on C++
classes.

Using the following C++ class :code:`Array` as example,

.. code:: c++

    template <typename T>
    class [[vineyard]] Array {
      public:
        [[shared]] const T& operator[](size_t loc) const { return data()[loc]; }
        [[shared]] size_t size() const { return size_; }
        [[shared]] const T* data() const {
            return reinterpret_cast<const T*>(buffer_->data());
        }

      private:
        [[shared]] size_t size_;
        [[shared]] std::shared_ptr<Blob> buffer_;
    };

- When applied on classes: the class itself will be identified as shared vineyard
  objects, a builder and resolver (see also :ref:`builder-resolver`) will be
  synthesized.

- When applied on data members: the data member will be treated as a metadata
  field or sub members.

- When applied on method members: the method member will be considered as
  cross-language sharable and FFI wrappers will be automatically synthesized.
