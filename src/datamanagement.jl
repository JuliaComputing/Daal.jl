module DataManagement

    using Cxx

    const doAllocateNumericTable   = icxx"daal::data_management::DataSource::doAllocateNumericTable;"
    const notAllocateNumericTable  = icxx"daal::data_management::DataSource::notAllocateNumericTable;"
    const doDictionaryFromContext  = icxx"daal::data_management::DataSource::doDictionaryFromContext;"
    const notDictionaryFromContext = icxx"daal::data_management::DataSource::notDictionaryFromContext;"

    FileDataSource(fileName::String,
                   doAllocateNumericTable = notAllocateNumericTable,
                   doCreateDictionaryFromContext = notDictionaryFromContext,
                   initialMaxRows = 10) = icxx"""
                        daal::data_management::FileDataSource<daal::data_management::CSVFeatureManager>($fileName, $doAllocateNumericTable, $doCreateDictionaryFromContext, $initialMaxRows);
                        """

    loadDataBlock(dataSource, nVectors::Integer) = icxx"$(dataSource).loadDataBlock($nVectors);"
    loadDataBlock(dataSource)                    = icxx"$(dataSource).loadDataBlock();"

    struct NumericTable
        o
    end

    getNumericTable(dataSource) = NumericTable(
        icxx"""
daal::services::SharedPtr<daal::data_management::NumericTable> result = $(dataSource).getNumericTable();
result;
            """)

    struct Tensor
        o::Cxx.CppValue
    end
end