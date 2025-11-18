.. _vcdl:

Code Generation for Boilerplate
===============================

The process of sharing objects between engines involves two fundamental steps: defining
the data structure and establishing the protocol to represent the data type
as :ref:`vineyard-objects`. To alleviate the integration burden with custom data types,
Vineyard offers an auto-generation mechanism.

This mechanism, known as **VCDL**, relies on the custom annotation :code:`[[shared]]`
applied to C++ classes.

Consider the following C++ class :code:`Array` as an example:

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

- When applied to classes: the class itself is identified as a shared vineyard
  object, and a builder and resolver (see also :ref:`builder-resolver`) are
  automatically synthesized.

- When applied to data members: the data member is treated as a metadata
  field or a sub-member.

- When applied to method members: the method member is deemed
  cross-language sharable, and FFI wrappers are automatically synthesized.
